"""DPT-style segmentation head.

The public ``SegHead`` API stays compatible with the W12 contract: it consumes
Swin stage features ``s1`` ... ``s4`` and returns full-resolution logits plus
an optional cross-entropy loss.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTaskHead


def _make_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _make_groups(channels)
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ReassembleBlock(nn.Module):
    """Project a Swin stage to the common DPT fusion width."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_make_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FusionBlock(nn.Module):
    """DPT refinement fusion block.

    ``x`` is the coarser feature. ``skip`` is the same-resolution feature from
    the next shallower Swin stage. The block refines, adds the skip connection,
    and upsamples when requested by the caller.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.res_skip = ResidualConvUnit(channels)
        self.res_out = ResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        out_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res_skip(skip)
        x = self.res_out(x)
        if out_size is not None and x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class SegHead(BaseTaskHead):
    task = "seg"
    in_channels: List[Optional[int]] = [96, 192, 384, 768]

    def __init__(
        self,
        in_chs: tuple[int, int, int, int] = (96, 192, 384, 768),
        num_classes: int = 4,
        dim: int = 256,
        input_size: int = 384,
    ):
        super().__init__()
        self.in_channels = list(in_chs)
        self.num_classes = num_classes
        self.input_size = input_size

        self.reassemble = nn.ModuleList([ReassembleBlock(c, dim) for c in in_chs])
        self.fuse4 = FusionBlock(dim)
        self.fuse3 = FusionBlock(dim)
        self.fuse2 = FusionBlock(dim)
        self.fuse1 = FusionBlock(dim)
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_make_groups(dim), dim),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(dim, num_classes, kernel_size=1),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stages = [feats[f"s{i}"] for i in range(1, 5)]
        s1, s2, s3, s4 = [proj(feat) for proj, feat in zip(self.reassemble, stages)]

        x = self.fuse4(s4, out_size=s3.shape[-2:])
        x = self.fuse3(x, s3, out_size=s2.shape[-2:])
        x = self.fuse2(x, s2, out_size=s1.shape[-2:])
        x = self.fuse1(x, s1)

        logits = self.head(x)
        out_hw = targets["mask"].shape[-2:] if targets is not None and "mask" in targets else None
        if out_hw is None:
            out_hw = (self.input_size, self.input_size)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)

        out = {"pred": logits}
        if targets is not None and "mask" in targets:
            loss = F.cross_entropy(logits, targets["mask"].long())
            out["loss"] = loss
            out["loss_items"] = {"seg/ce": float(loss.detach().cpu())}
        return out
