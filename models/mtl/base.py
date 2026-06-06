from __future__ import annotations

from abc import ABC
from typing import Dict

import torch.nn as nn


class MTLModel(nn.Module, ABC):
    def __init__(self, backbone: nn.Module, heads: Dict[str, nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, Dict]:
        outputs = {}
        for task, task_batch in batch.items():
            feats = self.backbone(task_batch["image"], task=task)
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))
        return outputs
