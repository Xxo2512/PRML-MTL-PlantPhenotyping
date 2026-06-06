"""分割 head (FPN-lite). W14 替换为完整 DPT。

数学:
  s4 (1/32 分辨率) 经 1x1 conv -> 上采样到 1/4
  与 s1 拼接, 再 conv -> 上采样到 1/1
  per-pixel cross entropy loss
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseTaskHead


def _conv(c_in, c_out, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=p),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class SegHead(BaseTaskHead):
    task = 'seg'
    in_channels: List[Optional[int]] = [96, 192, 384, 768]

    def __init__(self, in_chs=(96, 192, 384, 768), num_classes: int = 4, dim: int = 128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_chs])
        self.fuse = _conv(dim * 4, dim)
        self.cls_conv = nn.Conv2d(dim, num_classes, 1)

    def forward(self, feats: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]] = None):
        s = [feats[f's{i+1}'] for i in range(4)]
        H, W = s[0].shape[-2:]
        upsampled = []
        for i, f in enumerate(s):
            x = self.lat[i](f)
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            upsampled.append(x)
        x = self.fuse(torch.cat(upsampled, dim=1))
        logits = self.cls_conv(x)
        # 384x384 监督, 上采样到输入尺寸
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        out = {'pred': logits}
        if targets is not None and 'mask' in targets:
            loss = F.cross_entropy(logits, targets['mask'])
            out['loss'] = loss
            out['loss_items'] = {'seg/ce': loss.detach().item()}
        return out
