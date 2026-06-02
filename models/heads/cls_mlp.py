"""分类 head (生育期 6 类). 数学: y = softmax(W · GAP(f_s4) + b)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseTaskHead


class ClsHead(BaseTaskHead):
    task = 'cls'
    in_channels: List[Optional[int]] = [None, None, None, 768]

    def __init__(self, in_dim: int = 768, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]] = None):
        x = feats['s4']                     # [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # [B, C]
        x = self.norm(x)
        logits = self.fc(self.drop(x))               # [B, num_classes]
        out = {'pred': logits}
        if targets is not None and 'label' in targets:
            loss = F.cross_entropy(logits, targets['label'])
            out['loss'] = loss
            out['loss_items'] = {'cls/ce': loss.detach().item()}
        return out
