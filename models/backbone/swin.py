"""共享 Swin-T 骨干 (timm), 输出 4 个 stage 的 feature map。

数学:
  Swin Transformer (Liu et al. ICCV'21): 分层 patch + window self-attn,
  每 stage 通过 patch merging 降一半空间分辨率, 通道翻倍。
  swin_tiny: dims = [96, 192, 384, 768], strides = [4, 8, 16, 32]。
"""
from __future__ import annotations
import os
# 国内网络下默认走 hf-mirror, 不影响已设环境变量的用户
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from typing import Dict, Optional
import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    out_channels = [96, 192, 384, 768]
    out_strides  = [4, 8, 16, 32]

    def __init__(self, name: str = 'swin_tiny_patch4_window7_224',
                 pretrained: bool = True, img_size: int = 384):
        super().__init__()
        # timm: features_only=True 输出多尺度 feature
        self.model = timm.create_model(
            name, pretrained=pretrained, features_only=True,
            img_size=img_size, out_indices=(0, 1, 2, 3),
        )

    def forward(self, x: torch.Tensor, task: Optional[str] = None) -> Dict[str, torch.Tensor]:
        feats = self.model(x)               # 4 个 [B, H, W, C] tensor
        # timm Swin features_only 默认输出 NHWC, 需转成 NCHW 以便后续 head
        out = {}
        for i, f in enumerate(feats):
            if f.dim() == 4 and f.shape[-1] in self.out_channels:   # NHWC
                f = f.permute(0, 3, 1, 2).contiguous()
            out[f's{i+1}'] = f
        return out
