"""检测 head (W13 占位简化版, W14 替换为完整 FCOS)。

数学（简化）:
  在 stride=8 的 feature 上预测:
    obj : [B, 1, H, W]   sigmoid -> 是否目标中心
    reg : [B, 4, H, W]   ltrb 距离 (相对于该位置)
  loss = BCE(obj, gt_center_heatmap) + L1(reg, gt_ltrb) on positive locations
  gt_center_heatmap[y, x] = 1 if (x, y) 是某个 box 中心, else 0
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseTaskHead


class DetHead(BaseTaskHead):
    task = 'det'
    in_channels: List[Optional[int]] = [None, 192, None, None]   # 用 s2 (stride 8)

    def __init__(self, in_dim: int = 192, dim: int = 128, stride: int = 8, input_size: int = 384):
        super().__init__()
        self.stride = stride
        self.input_size = input_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
        )
        self.obj = nn.Conv2d(dim, 1, 1)
        self.reg = nn.Conv2d(dim, 4, 1)

    def _build_targets(self, boxes_list, H, W, device):
        """boxes: list of [N_i, 4] in normalized [0,1] xyxy。返回 obj_map[B,1,H,W] 与 reg_map[B,4,H,W]。"""
        B = len(boxes_list)
        obj = torch.zeros(B, 1, H, W, device=device)
        reg = torch.zeros(B, 4, H, W, device=device)
        for b, boxes in enumerate(boxes_list):
            if boxes.numel() == 0:
                continue
            # 转到 feature 坐标
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2 * W).clamp(0, W - 1)
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2 * H).clamp(0, H - 1)
            xs = cx.long(); ys = cy.long()
            obj[b, 0, ys, xs] = 1.0
            # ltrb in feature units
            l = (cx - boxes[:, 0] * W)
            t = (cy - boxes[:, 1] * H)
            r = (boxes[:, 2] * W - cx)
            d = (boxes[:, 3] * H - cy)
            for k in range(boxes.size(0)):
                reg[b, 0, ys[k], xs[k]] = l[k]
                reg[b, 1, ys[k], xs[k]] = t[k]
                reg[b, 2, ys[k], xs[k]] = r[k]
                reg[b, 3, ys[k], xs[k]] = d[k]
        return obj, reg

    def forward(self, feats: Dict[str, torch.Tensor], targets: Optional[Dict[str, Any]] = None):
        f = self.conv(feats['s2'])
        obj_logit = self.obj(f)
        reg = F.relu(self.reg(f))
        out = {'pred': {'obj_logit': obj_logit, 'reg': reg}}
        if targets is not None and 'boxes' in targets:
            B, _, H, W = obj_logit.shape
            boxes_list = [b.to(obj_logit.device) for b in targets['boxes']]
            obj_gt, reg_gt = self._build_targets(boxes_list, H, W, obj_logit.device)
            obj_loss = F.binary_cross_entropy_with_logits(obj_logit, obj_gt)
            mask = obj_gt > 0
            if mask.any():
                reg_loss = F.l1_loss(reg[mask.expand_as(reg)], reg_gt[mask.expand_as(reg_gt)])
            else:
                reg_loss = obj_logit.new_zeros(())
            loss = obj_loss + 0.1 * reg_loss
            out['loss'] = loss
            out['loss_items'] = {'det/obj': obj_loss.detach().item(),
                                 'det/reg': reg_loss.detach().item()}
        return out
