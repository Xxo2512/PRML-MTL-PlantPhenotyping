from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .base import MTLModel


class StageModulator(nn.Module):
    def __init__(self, task_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_dim, task_dim),
            nn.ReLU(inplace=True),
            nn.Linear(task_dim, channels * 2),
        )

    def forward(self, feat: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(task_embedding)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        return feat * (1.0 + gamma) + beta


class TADFormerModel(MTLModel):
    """Task-adaptive dynamic modulation wrapper.

    The full paper applies dynamic routing inside transformer blocks. This
    project skeleton keeps the frozen interface while applying per-task,
    per-stage feature modulation after the shared backbone, which is enough to
    connect dataloaders, heads, and train-loop experiments before deeper Swin
    block integration lands.
    """

    def __init__(
        self,
        cfg: Dict,
        backbone: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(cfg=cfg, backbone=backbone, heads=heads)
        model_cfg = self._get(cfg, "model", {})
        tad_cfg = self._get(model_cfg, "tadformer", {})
        self.tasks = tuple(
            task for task in ("seg", "det", "cnt", "cls") if self._get(cfg, f"tasks.{task}.enabled", True)
        )
        self.task_dim = int(self._get(tad_cfg, "task_dim", 64))
        self.per_stage = bool(self._get(tad_cfg, "per_stage", True))

        out_channels: Iterable[int] = getattr(self.backbone, "out_channels", [96, 192, 384, 768])
        self.task_embeddings = nn.ModuleDict(
            {task: nn.Embedding(1, self.task_dim) for task in self.tasks}
        )
        self.task_modules = nn.ModuleDict(
            {
                task: nn.ModuleDict(
                    {
                        f"s{idx + 1}": StageModulator(self.task_dim, channels)
                        for idx, channels in enumerate(out_channels)
                    }
                )
                for task in self.tasks
            }
        )
        if self._get(model_cfg, "freeze_backbone", False):
            self._freeze_backbone()

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, Dict]:
        outputs = {}
        for task, task_batch in batch.items():
            feats = self.backbone(task_batch["image"], task=task)
            feats = self._modulate(task, feats)
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))
        return outputs

    def _modulate(self, task: str, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        emb = self.task_embeddings[task](torch.zeros(1, dtype=torch.long, device=next(self.parameters()).device))
        modules = self.task_modules[task]
        return {key: modules[key](feat, emb) if key in modules else feat for key, feat in feats.items()}

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def _get(obj, path: str, default=None):
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = getattr(cur, key, default)
            if cur is default:
                break
        return cur
