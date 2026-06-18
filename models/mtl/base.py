from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

import torch.nn as nn


class MTLModel(nn.Module, ABC):
    def __init__(
        self,
        cfg=None,
        backbone: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        if backbone is None or heads is None:
            if cfg is None:
                raise TypeError("MTLModel requires either cfg or explicit backbone and heads")
            from ..backbone import SwinBackbone
            from ..heads import build_head

            self.backbone = SwinBackbone(
                name=cfg.model.backbone_full_name,
                pretrained=cfg.model.pretrained_flag,
                img_size=cfg.model.input_size,
            )
            enabled_tasks = [
                task for task in ("seg", "det", "cnt", "cls") if cfg.tasks[task].enabled
            ]
            self.heads = nn.ModuleDict({task: build_head(task, cfg) for task in enabled_tasks})
            if cfg.model.freeze_backbone:
                self._freeze_backbone()
        else:
            self.backbone = backbone
            self.heads = nn.ModuleDict(heads)

    def forward(self, batch: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        outputs = {}
        for task, task_batch in batch.items():
            if task not in self.heads:
                continue
            feats = self.backbone(task_batch["image"], task=task)
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))
        return outputs

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


class VanillaMTL(MTLModel):
    """Shared backbone with independent task heads."""


def build_mtl_model(cfg) -> MTLModel:
    if cfg.method == "vanilla":
        return VanillaMTL(cfg=cfg)
    if cfg.method == "mtlora":
        from .mtlora import MTLoRAModel

        return MTLoRAModel(cfg=cfg)
    if cfg.method == "tadformer":
        from .tadformer import TADFormerModel

        return TADFormerModel(cfg=cfg)
    if cfg.method == "pgt":
        from .pgt import PGTModel

        return PGTModel(cfg=cfg)
    if cfg.method == "ditask":
        from .ditask import DiTASKModel

        return DiTASKModel(cfg=cfg)
    if cfg.method == "taskprompter":
        from .taskprompter import TaskPrompterModel

        return TaskPrompterModel(cfg=cfg)
    raise NotImplementedError(f"method {cfg.method!r} not yet implemented")
