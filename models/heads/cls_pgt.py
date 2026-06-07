"""PGT 分类 head — 支持 Cross-Entropy 与 Ordinal Regression 两种损失。

Ordinal Regression 参考:
  Niu et al., "Ordinal Regression with Multiple Output CNN for Age Estimation",
  CVPR 2016.

数学原理 (K 类有序分类):
  - 构建 K-1 个二分类器, 第 k 个预测 P(y > k)
  - 目标编码: 对于类别 c, 二值目标 = [1 if c > k else 0 for k = 0..K-2]
  - 推理: class = Σ_k 1[sigmoid(logit_k) > 0.5]
  - Loss: (1/(K-1)) * Σ_k BCE(logit_k, target_k)

生育期 6 类天然有序: Tillering < Jointing < BH < Flowering < Filling < Ripening
相邻类误分 (如 Jointing→BH) 应轻于跨类误分 (如 Tillering→Ripening)。
Ordinal Regression 通过 K-1 个共享权重的二分类器隐式编码这种序关系。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTaskHead


class ClsPGTHead(BaseTaskHead):
    """PGT 分类头 — 支持 CE loss / Ordinal Regression loss.

    通过 yaml 配置 `tasks.cls.loss` 切换:
      - "ce": 标准 Cross-Entropy (默认)
      - "ordinal": Ordinal Regression (Niu et al., CVPR 2016)

    Args:
        in_dim: 输入特征维度 (来自 backbone stage 4: Swin-T = 768).
        num_classes: 生育期类别数 (默认 6).
        dropout: Dropout 比率.
        loss_type: 损失类型, "ce" 或 "ordinal".
    """

    task: str = "cls"
    in_channels: List[Optional[int]] = [None, None, None, 768]

    def __init__(
        self,
        in_dim: int = 768,
        num_classes: int = 6,
        dropout: float = 0.1,
        loss_type: str = "ce",
    ):
        super().__init__()
        if loss_type not in ("ce", "ordinal"):
            raise ValueError(f"loss_type must be 'ce' or 'ordinal', got {loss_type!r}")

        self.num_classes = num_classes
        self.loss_type = loss_type

        # Ordinal regression: K-1 个二分类输出
        if loss_type == "ordinal":
            self.num_ordinal_outputs = num_classes - 1
            out_dim = self.num_ordinal_outputs
        else:
            out_dim = num_classes

        # 与 cls_mlp.py 保持一致的 head 结构, 便于对比
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """前向传播.

        Args:
            feats: Backbone 多尺度特征, 含 's1'..'s4'.
                   cls head 仅消费 's4' [B, 768, 12, 12].
            targets: 训练时提供, 含 'label': [B] long tensor.

        Returns:
            Dict:
              - train: {'pred': logits, 'loss': scalar, 'loss_items': {key: float}}
              - eval:  {'pred': logits}
        """
        x = feats["s4"]                                      # [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)          # [B, C]
        x = self.norm(x)
        logits = self.fc(self.drop(x))                       # [B, out_dim]

        out: Dict[str, Any] = {"pred": logits}

        if targets is not None and "label" in targets:
            labels = targets["label"]                         # [B]

            if self.loss_type == "ordinal":
                loss = self._ordinal_loss(logits, labels)
                out["loss_items"] = {"cls/ordinal": loss.detach().item()}
            else:
                loss = F.cross_entropy(logits, labels)
                out["loss_items"] = {"cls/ce": loss.detach().item()}

            out["loss"] = loss

        return out

    # ------------------------------------------------------------------
    # Ordinal Regression Loss
    # ------------------------------------------------------------------

    def _ordinal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Ordinal Regression 损失 (Niu et al., CVPR 2016).

        将 K 类有序分类转化为 K-1 个独立二分类问题:
          target[k] = 1 如果 label > k, 否则 0   (k = 0, ..., K-2)

        总损失为 K-1 个 BCE 的均值:
          L = (1/(K-1)) * Σ_k BCEWithLogitsLoss(logits[:, k], target[:, k])

        Args:
            logits: [B, K-1] 原始 logits.
            labels: [B] 整数类别标签, 范围 [0, K-1].

        Returns:
            标量 loss tensor.
        """
        K = self.num_classes
        device = logits.device

        # 构造二值目标: [B, K-1]
        # target[k] = 1 当 label > k
        k_values = torch.arange(K - 1, device=device).unsqueeze(0)   # [1, K-1]
        labels_expanded = labels.unsqueeze(1)                         # [B, 1]
        binary_targets = (labels_expanded > k_values).float()         # [B, K-1]

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        return loss

    # ------------------------------------------------------------------
    # 推理辅助
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """将 logits 转换为类别预测.

        CE 模式: argmax.
        Ordinal 模式: 统计 sigmoid(logit) > 0.5 的二分类器数量.

        Args:
            logits: [B, out_dim] 模型输出.

        Returns:
            [B] 预测类别索引 (0 .. num_classes-1).
        """
        if self.loss_type == "ordinal":
            probs = torch.sigmoid(logits)                    # [B, K-1]
            preds = (probs > 0.5).sum(dim=1)                 # [B]
        else:
            preds = logits.argmax(dim=1)                     # [B]
        return preds

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """获取类别概率分布.

        CE 模式: softmax.
        Ordinal 模式: 从 K-1 个 P(y > k) 推导 P(y = k).
          P(y = 0) = 1 - P(y > 0)
          P(y = k) = P(y > k-1) - P(y > k)   (1 <= k <= K-2)
          P(y = K-1) = P(y > K-2)

        Args:
            logits: [B, out_dim] 模型输出.

        Returns:
            [B, num_classes] 概率分布, 每行和为 1.
        """
        if self.loss_type == "ordinal":
            probs_greater = torch.sigmoid(logits)            # [B, K-1], P(y > k)
            K = self.num_classes
            B = logits.shape[0]
            prob = torch.zeros(B, K, device=logits.device)
            prob[:, 0] = 1.0 - probs_greater[:, 0]           # P(y = 0)
            for k in range(1, K - 1):
                prob[:, k] = probs_greater[:, k - 1] - probs_greater[:, k]
            prob[:, K - 1] = probs_greater[:, K - 2]         # P(y = K-1)
            # 数值稳定: clamp 并重归一化
            prob = prob.clamp(min=1e-7)
            prob = prob / prob.sum(dim=1, keepdim=True)
            return prob
        else:
            return F.softmax(logits, dim=1)
