"""MTL 模型抽象基类 + vanilla 实现。

vanilla 数学:
  L_total = Σ_t L_t (任务出现在当前 batch 时)
  forward 一个 batch dict {task: {image, targets}}
"""
from __future__ import annotations
from abc import ABC
from typing import Dict, Any
import torch
import torch.nn as nn

from ..backbone import SwinBackbone
from ..heads import build_head


class MTLModel(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = SwinBackbone(
            name=cfg.model.backbone_full_name,
            pretrained=cfg.model.pretrained_flag,
            img_size=cfg.model.input_size,
        )
        enabled_tasks = [t for t in ('seg', 'det', 'cnt', 'cls') if cfg.tasks[t].enabled]
        self.heads = nn.ModuleDict({t: build_head(t, cfg) for t in enabled_tasks})
        if cfg.model.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, batch: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for task, t_batch in batch.items():
            if task not in self.heads:
                continue
            x = t_batch['image']
            feats = self.backbone(x, task=task)
            head_out = self.heads[task](feats, t_batch.get('targets'))
            out[task] = head_out
        return out


class VanillaMTL(MTLModel):
    """无任何 MTL trick: 只共享 backbone, 各 task head 独立。"""


def build_mtl_model(cfg) -> MTLModel:
    method = cfg.method
    if method == 'vanilla':
        return VanillaMTL(cfg)
    # W14 起填充其它方法:
    # if method == 'mtlora':       from .mtlora import MTLoRAModel;    return MTLoRAModel(cfg)
    # if method == 'tadformer':    ...
    raise NotImplementedError(f'method {method!r} not yet implemented')
