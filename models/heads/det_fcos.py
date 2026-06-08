from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTaskHead


class FCOSDetectionHead(BaseTaskHead):
    """A compact FCOS-style head for W13 smoke training."""

    task = "det"

    def __init__(
        self,
        in_channels: Optional[List[Optional[int]]] = None,
        num_classes: int = 1,
        feat_key: str = "s3",
        hidden_dim: int = 128,
        score_thresh: float = 0.05,
        topk: int = 100,
        in_dim: Optional[int] = None,
        stride: Optional[int] = None,
        input_size: int = 384,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            hidden_dim = dim
        self.in_channels = in_channels or [96, 192, 384, 768]
        self.num_classes = num_classes
        self.feat_key = feat_key
        self.score_thresh = score_thresh
        self.topk = topk
        self.input_size = input_size

        stage_index = int(feat_key[-1]) - 1
        channels = in_dim if in_dim is not None else self.in_channels[stage_index]
        if channels is None:
            raise ValueError(f"{feat_key} channel cannot be None for FCOSDetectionHead")

        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1)
        self.bbox_reg = nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, feats: Dict[str, torch.Tensor], targets=None) -> Dict[str, Any]:
        feat = feats[self.feat_key]
        hidden = self.stem(feat)
        cls_logits = self.cls_logits(hidden)
        bbox_reg = F.softplus(self.bbox_reg(hidden))
        centerness = self.centerness(hidden)

        pred = self._decode(cls_logits, bbox_reg, centerness)
        if self.training and targets is not None:
            loss, loss_items = self._loss(cls_logits, bbox_reg, centerness, targets)
            return {"loss": loss, "loss_items": loss_items, "pred": pred}
        return {"pred": pred, "metric": {}}

    def _loss(
        self,
        cls_logits: torch.Tensor,
        bbox_reg: torch.Tensor,
        centerness: torch.Tensor,
        targets: Dict,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        batch_size, _, height, width = cls_logits.shape
        device = cls_logits.device
        obj_target = torch.zeros((batch_size, 1, height, width), device=device)
        reg_target = torch.zeros((batch_size, 4, height, width), device=device)
        reg_mask = torch.zeros((batch_size, 1, height, width), device=device)

        stride_y = self.input_size / height
        stride_x = self.input_size / width
        boxes_per_image = targets["boxes"]
        for b_idx, boxes in enumerate(boxes_per_image):
            boxes = boxes.to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                continue
            if boxes.max() <= 1.5:
                boxes = boxes * self.input_size
            centers_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / stride_x).long().clamp(0, width - 1)
            centers_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / stride_y).long().clamp(0, height - 1)
            for box, cx, cy in zip(boxes, centers_x, centers_y):
                px = (cx.float() + 0.5) * stride_x
                py = (cy.float() + 0.5) * stride_y
                reg_target[b_idx, :, cy, cx] = torch.stack(
                    [px - box[0], py - box[1], box[2] - px, box[3] - py]
                ).clamp(min=0.0)
                obj_target[b_idx, :, cy, cx] = 1.0
                reg_mask[b_idx, :, cy, cx] = 1.0

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits[:, :1], obj_target)
        center_loss = F.binary_cross_entropy_with_logits(centerness, obj_target)
        if reg_mask.sum() > 0:
            reg_loss = F.l1_loss(bbox_reg * reg_mask, reg_target * reg_mask, reduction="sum")
            reg_loss = reg_loss / reg_mask.sum().clamp_min(1.0)
        else:
            reg_loss = bbox_reg.sum() * 0.0
        loss = cls_loss + reg_loss + 0.5 * center_loss
        return (
            loss,
            {
                "det/cls": float(cls_loss.detach().cpu()),
                "det/reg": float(reg_loss.detach().cpu()),
                "det/centerness": float(center_loss.detach().cpu()),
            },
        )

    def _decode(
        self,
        cls_logits: torch.Tensor,
        bbox_reg: torch.Tensor,
        centerness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = cls_logits.shape
        device = cls_logits.device
        scores = (cls_logits[:, :1].sigmoid() * centerness.sigmoid()).flatten(1)
        k = min(self.topk, scores.shape[1])
        top_scores, top_idx = scores.topk(k, dim=1)

        ys = torch.div(top_idx, width, rounding_mode="floor").float()
        xs = (top_idx % width).float()
        stride_y = self.input_size / height
        stride_x = self.input_size / width
        px = (xs + 0.5) * stride_x
        py = (ys + 0.5) * stride_y

        reg = bbox_reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        reg = torch.gather(reg, 1, top_idx.unsqueeze(-1).expand(-1, -1, 4))
        boxes = torch.stack(
            [px - reg[..., 0], py - reg[..., 1], px + reg[..., 2], py + reg[..., 3]],
            dim=-1,
        ).clamp(min=0.0, max=float(self.input_size))
        labels = torch.ones_like(top_scores, dtype=torch.long, device=device)
        return {"boxes": boxes, "scores": top_scores, "labels": labels}


class DetHead(FCOSDetectionHead):
    """Main-line compatible detection head name."""

    in_channels: List[Optional[int]] = [96, 192, 384, 768]
