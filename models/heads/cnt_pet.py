"""计数 head (W13: density regression from point targets; W14: 替换为 PET point-query).

数据流:
  feats['s2']  ->  Conv stack  ->  ReLU  ->  D_pred ∈ R^{B, 1, H', W'}
  count_pred = sum(D_pred over H', W')

训练目标 (从 targets['points'] 现算, head 持有 stride / sigma):
  对每个 batch sample 的点 (x, y) (像素 [0, 384)):
    D_gt[round(y/stride), round(x/stride)] += 1
  再做 σ=2 的 separable 高斯平滑.
  L = 1000 · MSE(D_pred, D_gt) + 0.01 · L1(count_pred, count_gt)

预测里的 'density' 是 head 内部表示, 仅用于可视化; 契约暴露的核心 pred 是 'count'.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTaskHead


def _gaussian_kernel1d(sigma: float, *, device, dtype) -> torch.Tensor:
    radius = max(1, int(round(4.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _points_to_density(
    points_list: List[torch.Tensor],
    H: int,
    W: int,
    *,
    stride: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """list of [N_i, 2] 像素坐标 -> [B, 1, H, W] density (高斯平滑后)."""
    B = len(points_list)
    density = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    for b, pts in enumerate(points_list):
        if pts.numel() == 0:
            continue
        p = pts.to(device=device, dtype=dtype) / stride
        xs = p[:, 0].round().long().clamp(0, W - 1)
        ys = p[:, 1].round().long().clamp(0, H - 1)
        ones = torch.ones_like(xs, dtype=dtype)
        density[b, 0].index_put_((ys, xs), ones, accumulate=True)
    if sigma > 0:
        k = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
        pad = k.shape[0] // 2
        density = F.conv2d(density, k.view(1, 1, 1, -1), padding=(0, pad))
        density = F.conv2d(density, k.view(1, 1, -1, 1), padding=(pad, 0))
    return density


class CntHead(BaseTaskHead):
    task = 'cnt'
    in_channels: List[Optional[int]] = [None, 192, None, None]   # 用 s2 (stride 8)
    DENSITY_STRIDE = 8
    DENSITY_SIGMA  = 2.0

    def __init__(self, in_dim: int = 192, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, Any]] = None,
    ):
        density = F.relu(self.net(feats['s2']))     # [B, 1, H, W]
        count_pred = density.flatten(1).sum(1)      # [B]
        out = {'pred': {'density': density, 'count': count_pred}}
        if targets is not None and 'points' in targets:
            _, _, H, W = density.shape
            d_gt = _points_to_density(
                targets['points'], H, W,
                stride=self.DENSITY_STRIDE, sigma=self.DENSITY_SIGMA,
                device=density.device, dtype=density.dtype,
            )                                       # [B, 1, H, W]
            # Loss 平衡:
            # - density GT 是稀疏高斯峰 (~4% 像素非零), 全零预测能让普通 MSE 几乎为 0,
            #   旧实现 MSE×1000 + 0.01×L1(count) 让模型收敛到"输出全零"的退化解 (验证发现 3 个
            #   独立训练的 cnt 模型 val MAE 完全相同 = 训练集 mean count).
            # - 新平衡: 不放大 MSE; 加大 count L1 直接监督全图积分 (counting 任务的最终目标).
            mse = F.mse_loss(density, d_gt)
            cnt_l1 = F.l1_loss(count_pred, targets['count'].to(density.dtype))
            loss = mse + 1.0 * cnt_l1
            out['loss'] = loss
            out['loss_items'] = {
                'cnt/mse': mse.detach().item(),
                'cnt/l1':  cnt_l1.detach().item(),
            }
        return out
