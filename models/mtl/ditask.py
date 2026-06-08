from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .base import MTLModel


class LowRankDiffeomorphicMap(nn.Module):
    """Lightweight residual feature map used as DiTASK Phi_t.

    This first project implementation keeps the diffeomorphic fine-tuning idea
    in a stable, interface-compatible form: each stage learns a near-identity
    low-rank 1x1 residual transform, initialized so the pretrained backbone
    behavior is preserved at step 0.
    """

    def __init__(self, channels: int, rank: int, alpha_init: float = 1.0e-3):
        super().__init__()
        rank = max(1, min(int(rank), int(channels)))
        self.down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.down(x))


class DiTASKModel(MTLModel):
    """DiTASK wrapper applying task-specific Phi_t after backbone features."""

    def __init__(
        self,
        cfg: Dict,
        backbone: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__(cfg=cfg, backbone=backbone, heads=heads)
        model_cfg = self._get(cfg, "model", {})
        ditask_cfg = self._get(model_cfg, "ditask", {})
        self.phi_type = str(self._get(ditask_cfg, "phi_type", "svd_rotation"))
        if self.phi_type != "svd_rotation":
            raise ValueError(f"unsupported DiTASK phi_type {self.phi_type!r}")
        self.rank = int(self._get(ditask_cfg, "rank", 8))
        self.tasks = tuple(
            task for task in ("seg", "det", "cnt", "cls") if self._get(cfg, f"tasks.{task}.enabled", True)
        )

        out_channels: Iterable[int] = getattr(self.backbone, "out_channels", [96, 192, 384, 768])
        self.task_modules = nn.ModuleDict(
            {
                task: nn.ModuleDict(
                    {
                        f"s{idx + 1}": LowRankDiffeomorphicMap(channels, self.rank)
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
            if task not in self.heads:
                continue
            feats = self.backbone(task_batch["image"], task=task)
            feats = self._apply_phi(task, feats)
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))
        return outputs

    def _apply_phi(self, task: str, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if task not in self.task_modules:
            return feats
        modules = self.task_modules[task]
        return {key: modules[key](feat) if key in modules else feat for key, feat in feats.items()}

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
