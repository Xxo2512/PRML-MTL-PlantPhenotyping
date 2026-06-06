"""计数 head (W13 用 density regression, W14 替换为 PET point-query)。

数学:
  D_pred = head(f_s2)        ∈ R^{B,1,H,W}     (预测密度图)
  L = MSE(D_pred, D_gt)
  count_pred = sum(D_pred over H,W)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseTaskHead


class CntHead(BaseTaskHead):
    task = 'cnt'
    in_channels: List[Optional[int]] = [None, 192, None, None]   # 用 s2 (stride 8)

    def __init__(self, in_dim: int = 192, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(self, feats: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]] = None):
        density = F.relu(self.net(feats['s2']))     # [B, 1, H, W]
        count_pred = density.flatten(1).sum(1)      # [B]
        out = {'pred': {'density': density, 'count': count_pred}}
        if targets is not None and 'density' in targets:
            d_gt = targets['density'].unsqueeze(1)  # [B, 1, H, W]
            mse = F.mse_loss(density, d_gt) * 1000.0   # density 数量级很小, 放大
            cnt_l1 = F.l1_loss(count_pred, targets['count'])
            loss = mse + 0.01 * cnt_l1
            out['loss'] = loss
            out['loss_items'] = {'cnt/mse': mse.detach().item(),
                                 'cnt/l1':  cnt_l1.detach().item()}
        return out
