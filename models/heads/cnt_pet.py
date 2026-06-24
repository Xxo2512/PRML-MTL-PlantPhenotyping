"""PET (Point quEry Transformer, Liu et al. ICCV 2023) 风格的计数头.

实际范式而不是文件名表面: 不是 density regression, 是 point-query + Hungarian + CE/SmoothL1.
对照参考: https://github.com/cxliu0/PET

本实现是简化版 PET (不含 quadtree 自适应稀疏/密集分支):
  - N 个 learnable point queries (内容 + position 双 embedding)
  - 多层 nn.TransformerDecoder 让 queries cross-attend backbone features (s2, stride 8)
  - 每个 query 输出 (class_logit, x, y) — 二分类 fg/bg + 归一化坐标
  - 训练时: Hungarian matching 把 N 个 query 与 M 个 GT 点配对
      cls_loss = CE(matched=fg, unmatched=bg)
      reg_loss = SmoothL1(matched_pred_xy, matched_gt_xy)
      aux_cnt  = SmoothL1(sum(softmax(cls)[:, fg]), count_gt)  # 软计数辅助稳定训练
  - 推理时: count = #{queries with softmax(cls)[:, fg] > 0.5}
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .base import BaseTaskHead


class CntHead(BaseTaskHead):
    task = "cnt"
    in_channels: List[Optional[int]] = [None, 192, None, None]  # 使用 s2
    FEAT_STRIDE = 8                  # backbone s2 相对原图 stride
    INPUT_SIZE = 384                 # InputAdapter 输出尺寸, 用于坐标归一化

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 128,
        num_queries: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 5.0,
        aux_count_weight: float = 0.1,
        match_cls_weight: float = 1.0,
        match_reg_weight: float = 5.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.aux_count_weight = aux_count_weight
        self.match_cls_weight = match_cls_weight
        self.match_reg_weight = match_reg_weight

        # backbone feature → hidden
        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        # 2D 位置编码 (固定 sin-cos, 不学)
        # 实际生成放 forward 里 (依赖 H,W), 这里只造好维度匹配

        # Learnable queries (内容 + position 两个 embedding)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出 heads
        self.cls_head = nn.Linear(hidden_dim, 2)        # [bg, fg]
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),                   # (x, y), 后接 sigmoid 到 [0,1]
        )

        # 初始化: 把 coord_head 最后一层 bias 设成 0, 让初始预测在图中心附近
        nn.init.constant_(self.coord_head[-1].weight, 0.0)
        nn.init.constant_(self.coord_head[-1].bias, 0.0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        f = feats["s2"]                                  # [B, in_dim, H, W]
        B, _, H, W = f.shape
        f_proj = self.input_proj(f)                      # [B, hidden, H, W]
        pos = self._build_2d_sincos_pe(H, W, self.hidden_dim, device=f.device, dtype=f.dtype)  # [1, hidden, H, W]
        memory = (f_proj + pos).flatten(2).transpose(1, 2)   # [B, H*W, hidden]

        q_content = self.query_content.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, hidden]
        q_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = q_content + q_pos                          # 加位置编码进 query

        out = self.decoder(tgt, memory)                  # [B, N, hidden]
        cls_logit = self.cls_head(out)                   # [B, N, 2]
        coords = self.coord_head(out).sigmoid()          # [B, N, 2]  in [0,1]

        # 推理 count: 软计数 (训练用) + 硬计数 (诊断保留)
        # 注: 训练 aux loss 监督的是 soft_count, 所以评测也用 soft_count, 保持口径一致.
        # hard_count (threshold 0.5) 对欠训练模型偏低, 仅作诊断对照.
        cls_prob = cls_logit.softmax(dim=-1)             # [B, N, 2]
        soft_count = cls_prob[..., 1].sum(dim=-1)        # [B]  可微分软计数
        hard_count = (cls_prob[..., 1] > 0.5).sum(dim=-1).float()  # [B]  仅作诊断对照

        pred = {
            "logits": cls_logit,                         # [B, N, 2]
            "points": coords,                            # [B, N, 2] 归一化
            "count": soft_count,                         # eval 用这个 (与训练监督一致)
            "soft_count": soft_count,                    # 同 count, 保留兼容
            "hard_count": hard_count,                    # 诊断用 (threshold 0.5)
        }
        result: Dict[str, Any] = {"pred": pred}

        if targets is not None and "points" in targets:
            loss, items = self._loss(cls_logit, coords, soft_count, targets)
            result["loss"] = loss
            result["loss_items"] = items

        return result

    # ------------------------------------------------------------------
    # Loss: Hungarian match + CE + SmoothL1 + aux count
    # ------------------------------------------------------------------
    def _loss(
        self,
        cls_logit: torch.Tensor,                         # [B, N, 2]
        coords: torch.Tensor,                            # [B, N, 2] in [0,1]
        soft_count: torch.Tensor,                        # [B]
        targets: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = cls_logit.device
        B, N, _ = cls_logit.shape
        gt_points_list: List[torch.Tensor] = targets["points"]
        gt_count = targets["count"].to(device=device, dtype=cls_logit.dtype)

        cls_probs = cls_logit.softmax(dim=-1)            # [B, N, 2]

        cls_losses: List[torch.Tensor] = []
        reg_losses: List[torch.Tensor] = []
        n_matched = 0
        n_total_gt = 0

        for b in range(B):
            gt_pts_pix = gt_points_list[b].to(device=device, dtype=cls_logit.dtype)  # [M, 2] in pixel [0, 384)
            M = gt_pts_pix.shape[0]
            n_total_gt += M

            cls_b = cls_logit[b]                          # [N, 2]
            coord_b = coords[b]                           # [N, 2]

            if M == 0:
                # 全部 query 标为 bg
                tgt_cls = torch.zeros(N, dtype=torch.long, device=device)
                cls_losses.append(F.cross_entropy(cls_b, tgt_cls))
                continue

            gt_norm = gt_pts_pix / float(self.INPUT_SIZE)     # [M, 2] in [0,1]
            # ---- 构造 cost matrix [N, M] ----
            # 分类 cost: 越接近 fg 越 "便宜" → -fg_prob
            cost_cls = -cls_probs[b, :, 1:2].repeat(1, M)     # [N, M]
            # 坐标 cost: L1 距离
            cost_reg = torch.cdist(coord_b, gt_norm, p=1)     # [N, M]
            cost = self.match_cls_weight * cost_cls + self.match_reg_weight * cost_reg

            # Hungarian on CPU (scipy)
            with torch.no_grad():
                row_ind_np, col_ind_np = linear_sum_assignment(cost.detach().cpu().numpy())
            row_idx = torch.as_tensor(row_ind_np, device=device, dtype=torch.long)
            col_idx = torch.as_tensor(col_ind_np, device=device, dtype=torch.long)

            # ---- 分类 target: matched=fg, others=bg ----
            tgt_cls = torch.zeros(N, dtype=torch.long, device=device)
            tgt_cls[row_idx] = 1
            cls_losses.append(F.cross_entropy(cls_b, tgt_cls))

            # ---- 回归: 只在 matched 上做 SmoothL1 ----
            matched_pred = coord_b[row_idx]                   # [match, 2]
            matched_gt = gt_norm[col_idx]                     # [match, 2]
            reg_losses.append(F.smooth_l1_loss(matched_pred, matched_gt, reduction="mean"))
            n_matched += len(row_idx)

        cls_loss = torch.stack(cls_losses).mean() if cls_losses else cls_logit.sum() * 0.0
        reg_loss = (
            torch.stack(reg_losses).mean()
            if reg_losses
            else cls_logit.sum() * 0.0
        )
        aux_count = F.smooth_l1_loss(soft_count, gt_count)

        total = (
            self.cls_loss_weight * cls_loss
            + self.reg_loss_weight * reg_loss
            + self.aux_count_weight * aux_count
        )

        return total, {
            "cnt/cls": cls_loss.detach().item(),
            "cnt/reg": reg_loss.detach().item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss),
            "cnt/aux_count": aux_count.detach().item(),
            "cnt/n_matched_per_batch": n_matched / max(B, 1),
        }

    # ------------------------------------------------------------------
    # 2D sin-cos 位置编码 (固定, 不学习)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_2d_sincos_pe(H: int, W: int, hidden: int, *, device, dtype) -> torch.Tensor:
        """返回 [1, hidden, H, W] 的 2D sin-cos PE (DETR 风格简化)."""
        if hidden % 4 != 0:
            # 退化: 兜底用 0
            return torch.zeros(1, hidden, H, W, device=device, dtype=dtype)
        d = hidden // 2  # 给 x 与 y 各 d 维 → 总 2d = hidden
        omega = torch.arange(d // 2, device=device, dtype=dtype)
        omega = 1.0 / (10000.0 ** (2.0 * omega / d))

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)

        pe_y = y.unsqueeze(1) * omega.unsqueeze(0)        # [H, d/2]
        pe_x = x.unsqueeze(1) * omega.unsqueeze(0)        # [W, d/2]

        pe_y = torch.cat([pe_y.sin(), pe_y.cos()], dim=1) # [H, d]
        pe_x = torch.cat([pe_x.sin(), pe_x.cos()], dim=1) # [W, d]

        pe = torch.zeros(hidden, H, W, device=device, dtype=dtype)
        pe[:d, :, :] = pe_y.transpose(0, 1).unsqueeze(2).expand(d, H, W)
        pe[d:, :, :] = pe_x.transpose(0, 1).unsqueeze(1).expand(d, H, W)
        return pe.unsqueeze(0)
