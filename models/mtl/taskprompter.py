"""TaskPrompter: Spatial-Channel Multi-task Prompting (ICLR 2023).

参考: Ye & Xu, "TaskPrompter: Spatial-Channel Multi-task Prompting for
      Dense Scene Understanding", ICLR 2023.

核心思想:
  每个任务持有两类可学习 prompt，在共享 backbone 的每个 stage 注入:
  1. SPATIAL PROMPT — n_spatial 个可学习 token，通过 dot-product attention
     从图像特征中定位任务相关区域，产生空间注意力权重调制特征。
  2. CHANNEL PROMPT — 每通道 (γ, β) FiLM 参数，直接调制特征通道响应。

  两类 prompt 互补: spatial 定位"在哪里看"，channel 决定"看哪些特征"。

架构:
  1. 共享 Swin backbone 提取 4 级多尺度特征 {s1, s2, s3, s4}
  2. 特征投影 (stage_dim → prompt_dim)，共享于所有任务
  3. 空间注意力: prompt token (Q) × 特征 (K) → 空间权重 → 调制
  4. 输出投影 (prompt_dim → stage_dim)，残差连接回 backbone 特征
  5. 通道 FiLM: 在原始特征通道上施加 per-channel γ, β
  6. 调制后特征送入各自 task head

与项目框架的对接:
  - 继承 MTLModel, 不改父类签名
  - forward 接受 batch dict (支持 RR/PS/HM 三种调度)
  - backbone 冻结时仅训练 prompt + projection + head 参数
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MTLModel


# ══════════════════════════════════════════════════════════════════════
# 特征投影模块
# ══════════════════════════════════════════════════════════════════════


class FeatureProjection(nn.Module):
    """将 backbone 各 stage 特征投影到统一 prompt 维度.

    Conv1×1 + GroupNorm, 保证不同 stage 的特征在统一维度空间中交互.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = out_channels
        for g in (32, 16, 8, 4, 2, 1):
            if out_channels % g == 0:
                groups = g
                break
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════
# 空间提示注意力模块
# ══════════════════════════════════════════════════════════════════════


class SpatialPromptAttention(nn.Module):
    """空间提示注意力: prompt token 通过点积注意力定位任务相关区域.

    数学:
      Q = W_q · prompts          ∈ R^{B × n_spatial × D}
      K = W_k · feat_flat        ∈ R^{B × HW × D}
      A = softmax(Q K^T / √D)    ∈ R^{B × n_spatial × HW}
      w = mean_pool(A)           ∈ R^{B × HW}           (聚合所有 prompt)
      w = reshape(w) → [B, 1, H, W]
      feat' = feat ⊙ (1 + w)                              (空间调制)

    参数:
        prompt_dim: 统一投影维度.
    """

    def __init__(self, prompt_dim: int = 256):
        super().__init__()
        self.q_proj = nn.Linear(prompt_dim, prompt_dim, bias=False)
        self.k_proj = nn.Linear(prompt_dim, prompt_dim, bias=False)
        self.scale = prompt_dim ** -0.5
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

    def forward(
        self, prompts: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        """前向传播.

        Args:
            prompts: [B, n_spatial, D] — 可学习空间提示 token (已扩展 batch).
            feat:    [B, D, H, W]       — 投影后的图像特征.

        Returns:
            空间调制后的特征 [B, D, H, W].
        """
        B, D, H, W = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)            # [B, HW, D]

        Q = self.q_proj(prompts)                                # [B, n_spatial, D]
        K = self.k_proj(feat_flat)                              # [B, HW, D]

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale           # [B, n_spatial, HW]
        attn = attn.softmax(dim=-1)

        # 聚合所有 spatial prompt 的注意力 → 单一空间权重图
        spatial_weight = attn.mean(dim=1)                       # [B, HW]
        spatial_weight = spatial_weight.view(B, 1, H, W)        # [B, 1, H, W]

        return feat * (1.0 + spatial_weight)


# ══════════════════════════════════════════════════════════════════════
# 通道提示 FiLM 模块
# ══════════════════════════════════════════════════════════════════════


class ChannelPromptFiLM(nn.Module):
    """通道提示: 直接学习 per-channel FiLM 参数 (γ, β).

    数学:
      feat' = feat ⊙ (1 + γ) + β
      其中 γ, β ∈ R^C 是可学习参数.

    初始化为 γ=0, β=0 → 训练初期恒等, 不改变 backbone 特征.

    参数:
        channels: 特征通道数 (stage 原始维度, 非 prompt_dim).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            feat: [B, C, H, W] — backbone stage 原始特征 (已做完空间调制).

        Returns:
            通道调制后的特征 [B, C, H, W].
        """
        return feat * (1.0 + self.gamma) + self.beta


# ══════════════════════════════════════════════════════════════════════
# TaskPrompter MTL 模型主体
# ══════════════════════════════════════════════════════════════════════


class TaskPrompterModel(MTLModel):
    """TaskPrompter: Spatial-Channel Multi-task Prompting 多任务模型.

    设计要点:
      - 每个启用的 task 持有 per-stage 的空间提示 token 与通道 FiLM 参数
      - 空间提示通过 dot-product attention 产生空间注意力权重
      - 通道提示以 FiLM 方式直接调制每个通道
      - 特征投影与输出投影在所有任务间共享, 控制参数增量
      - Backbone 默认冻结, 仅训练 prompt + projection + head 参数
      - 完全兼容 train.py 与 evaluate.py 的接口

    Config (configs/method/taskprompter.yaml):
      model:
        freeze_backbone: true
        taskprompter:
          n_spatial: 4          # 空间提示 token 数量
          n_channel: 4          # 通道提示参数对数量 (预留, 当前隐含在 per-task FiLM 中)
          prompt_dim: 256       # 统一投影维度
    """

    def __init__(
        self,
        cfg=None,
        backbone: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None,
    ):
        # ---- 父类: 构建 backbone + heads ----
        super().__init__(cfg=cfg, backbone=backbone, heads=heads)

        # ---- 读取 TaskPrompter 专属配置 ----
        model_cfg = self._get_cfg(cfg, "model", {})
        tp_cfg = self._get_cfg(model_cfg, "taskprompter", {})

        self.n_spatial = int(self._get_cfg(tp_cfg, "n_spatial", 4))
        self.n_channel = int(self._get_cfg(tp_cfg, "n_channel", 4))
        self.prompt_dim = int(self._get_cfg(tp_cfg, "prompt_dim", 256))

        # ---- 确定启用的任务列表 ----
        self.tasks: Tuple[str, ...] = tuple(
            task
            for task in ("seg", "det", "cnt", "cls")
            if self._get_cfg(cfg, f"tasks.{task}.enabled", True)
        )
        print(f"[taskprompter] enabled tasks: {self.tasks}")

        # ---- Backbone 输出通道 ----
        out_channels: List[int] = getattr(
            self.backbone, "out_channels", [96, 192, 384, 768]
        )
        self.out_channels = out_channels
        stage_keys = [f"s{i + 1}" for i in range(len(out_channels))]
        self.stage_keys = stage_keys

        # ---- 1) 特征投影（所有任务共享）: stage_dim → prompt_dim ----
        self.feat_proj = nn.ModuleDict({
            stage_key: FeatureProjection(ch, self.prompt_dim)
            for stage_key, ch in zip(stage_keys, out_channels)
        })

        # ---- 2) 空间提示 token（每任务独立） ----
        #        形状: [1, n_spatial, prompt_dim], forward 时 expand 到 batch
        self.spatial_prompts = nn.ParameterDict()
        for task in self.tasks:
            self.spatial_prompts[task] = nn.Parameter(
                torch.randn(1, self.n_spatial, self.prompt_dim) * 0.02
            )

        # ---- 3) 空间注意力模块（所有任务共享） ----
        self.spatial_attn = nn.ModuleDict({
            stage_key: SpatialPromptAttention(self.prompt_dim)
            for stage_key in stage_keys
        })

        # ---- 4) 通道 FiLM（每任务 × stage 独立） ----
        self.channel_films = nn.ModuleDict()
        for task in self.tasks:
            self.channel_films[task] = nn.ModuleDict({
                stage_key: ChannelPromptFiLM(ch)
                for stage_key, ch in zip(stage_keys, out_channels)
            })

        # ---- 5) 输出投影（所有任务共享）: prompt_dim → stage_dim ----
        self.out_proj = nn.ModuleDict({
            stage_key: nn.Conv2d(self.prompt_dim, ch, kernel_size=1)
            for stage_key, ch in zip(stage_keys, out_channels)
        })
        # 零初始化输出投影 → 训练初期残差分支贡献为 0, 特征恒等
        for conv in self.out_proj.values():
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

        # ---- 冻结 backbone（按配置） ----
        if self._get_cfg(model_cfg, "freeze_backbone", False):
            self._freeze_backbone()

        # ---- 参数统计 ----
        self._log_params()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """多任务前向传播.

        流程 (每个 task 独立):
          1. Backbone 提取多尺度特征
          2. TaskPrompter 双提示调制:
             a. 特征投影 → spatial prompt attention → 反投影 → 残差
             b. Channel FiLM 调制
          3. Task head 消费调制后特征

        Args:
            batch: {'seg': {'image': ..., 'targets': ...}, ...}
                   RR/PS 模式下 len(batch) == 1; HM 模式下 len(batch) <= 4.

        Returns:
            {task: {'pred': ..., 'loss': ..., 'loss_items': ...}}
        """
        outputs: Dict[str, Dict[str, Any]] = {}

        for task, task_batch in batch.items():
            if task not in self.heads:
                continue

            # Step 1: Backbone 提取多尺度特征
            feats = self.backbone(task_batch["image"], task=task)

            # Step 2: TaskPrompter 双提示调制
            feats = self._apply_taskprompter(task, feats)

            # Step 3: Task head
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))

        return outputs

    # ------------------------------------------------------------------
    # TaskPrompter 核心: 双提示特征调制
    # ------------------------------------------------------------------

    def _apply_taskprompter(
        self, task: str, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """对 backbone 输出的每个 stage 特征施加空间 + 通道双提示调制.

        Pipeline (每个 stage 独立):
          ┌─────────────────────────────────────────────────┐
          │  feat ∈ R^{B×C×H×W}  (backbone 原始特征)       │
          │      │                                          │
          │      ├─ 1) 投影:  feat_proj → R^{B×D×H×W}       │
          │      │                                          │
          │      ├─ 2) 空间:  spatial_attn(prompts, feat)    │
          │      │           → 空间注意力权重调制             │
          │      │                                          │
          │      ├─ 3) 反投影: out_proj → R^{B×C×H×W}       │
          │      │                                          │
          │      ├─ 4) 残差:  feat = feat + out              │
          │      │                                          │
          │      └─ 5) 通道:  channel_films(feat)            │
          │                 → per-channel γ, β FiLM          │
          └─────────────────────────────────────────────────┘

        Args:
            task:  任务名.
            feats: {'s1': [B,96,96,96], 's2': [B,192,48,48],
                    's3': [B,384,24,24], 's4': [B,768,12,12]}

        Returns:
            调制后的特征 dict (相同 keys 和 shapes).
        """
        B = next(iter(feats.values())).shape[0]

        # 扩展空间提示 token: [1, n_spatial, D] → [B, n_spatial, D]
        prompts = self.spatial_prompts[task].expand(B, -1, -1)

        modulated: Dict[str, torch.Tensor] = {}
        for stage_key in self.stage_keys:
            feat = feats[stage_key]                               # [B, C, H, W]

            # 1+2) 投影 → 空间提示注意力
            feat_proj = self.feat_proj[stage_key](feat)           # [B, D, H, W]
            feat_spatial = self.spatial_attn[stage_key](
                prompts, feat_proj
            )                                                     # [B, D, H, W]

            # 3) 反投影回原始通道
            feat_out = self.out_proj[stage_key](feat_spatial)     # [B, C, H, W]

            # 4) 残差连接: 保留 backbone 原始特征
            feat = feat + feat_out

            # 5) 通道 FiLM 调制
            feat = self.channel_films[task][stage_key](feat)

            modulated[stage_key] = feat

        return modulated

    # ------------------------------------------------------------------
    # Loss & Metrics (兼容 evaluate.py)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """从 forward 输出中提取各任务 loss.

        Args:
            outputs: model(batch) 的返回值.

        Returns:
            {'cls/loss': tensor, 'seg/loss': tensor, ...}
        """
        losses: Dict[str, torch.Tensor] = {}
        for task, out in outputs.items():
            if "loss" in out:
                losses[f"{task}/loss"] = out["loss"]
        return losses

    def compute_metrics(
        self,
        outputs: Dict[str, Dict[str, Any]],
        batch: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """计算各任务评测指标.

        Args:
            outputs: model(batch) 的返回值.
            batch:   原始输入 batch (含 targets).

        Returns:
            Flat dict: {'cls/acc': 0.85, 'seg/mIoU': 0.67, ...}
        """
        from utils.metrics import (
            compute_cls_metric,
            compute_seg_metric,
            compute_det_metric,
            compute_cnt_metric,
        )

        all_metrics: Dict[str, float] = {}

        for task, out in outputs.items():
            if "pred" not in out:
                continue
            task_batch = batch.get(task, {})
            targets = task_batch.get("targets", {})

            if task == "cls":
                logits = out["pred"]
                labels = targets.get("label")
                if labels is not None:
                    head = self.heads.get("cls")
                    if hasattr(head, "predict_class"):
                        preds = head.predict_class(logits)
                    elif logits.shape[-1] > 1:
                        preds = logits.argmax(dim=-1)
                    else:
                        preds = (torch.sigmoid(logits) > 0.5).sum(dim=-1)
                    all_metrics.update(compute_cls_metric(logits, labels, preds))

            elif task == "seg":
                logits = out["pred"]
                mask = targets.get("mask")
                if mask is not None:
                    all_metrics.update(compute_seg_metric(logits, mask))

            elif task == "det":
                preds = out["pred"]
                boxes_gt = targets.get("boxes")
                labels_gt = targets.get("labels")
                if boxes_gt is not None:
                    all_metrics.update(
                        compute_det_metric(preds, boxes_gt, labels_gt)
                    )

            elif task == "cnt":
                if isinstance(out["pred"], dict):
                    count_pred = out["pred"].get("count")
                else:
                    count_pred = out["pred"]
                count_gt = targets.get("count")
                if count_pred is not None and count_gt is not None:
                    all_metrics.update(compute_cnt_metric(count_pred, count_gt))

        return all_metrics

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optimizer_params(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        lr_mult_backbone: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """返回分组优化器参数, 支持差异化学习率.

        分组策略:
          - Backbone: lr * lr_mult_backbone (如果未冻结)
          - TaskPrompter 专属参数 (prompt tokens, projections, attn, FiLM): lr
          - Head 参数: lr

        Args:
            lr: 基础学习率.
            weight_decay: 权重衰减.
            lr_mult_backbone: Backbone 学习率倍率.

        Returns:
            符合 torch.optim.AdamW params 格式的参数组列表.
        """
        param_groups: List[Dict[str, Any]] = []

        # Group 1: Backbone (如果可训练)
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": lr * lr_mult_backbone,
                "weight_decay": weight_decay,
            })

        # Group 2: TaskPrompter 专属参数
        tp_param_ids = set()
        for attr_name in [
            "feat_proj",
            "spatial_prompts",
            "spatial_attn",
            "channel_films",
            "out_proj",
        ]:
            if hasattr(self, attr_name):
                module = getattr(self, attr_name)
                for p in module.parameters():
                    if p.requires_grad:
                        tp_param_ids.add(id(p))
        if tp_param_ids:
            param_groups.append({
                "params": [
                    p
                    for p in self.parameters()
                    if id(p) in tp_param_ids and p.requires_grad
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            })

        # Group 3: Head 参数 (排除已归入 group 2 的)
        head_params = []
        for head in self.heads.values():
            for p in head.parameters():
                if p.requires_grad and id(p) not in tp_param_ids:
                    head_params.append(p)
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })

        return param_groups

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """冻结 backbone 全部参数."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """切换训练模式, 同时确保被冻结的 backbone 保持在 eval 模式."""
        super().train(mode)
        if mode:
            model_cfg = self._get_cfg(self.cfg, "model", {})
            if self._get_cfg(model_cfg, "freeze_backbone", False):
                self.backbone.eval()
        return self

    def _log_params(self) -> None:
        """打印 TaskPrompter 参数统计."""
        # 统计各类参数
        n_tp = sum(
            p.numel()
            for attr in ["feat_proj", "spatial_prompts", "spatial_attn",
                         "channel_films", "out_proj"]
            if hasattr(self, attr)
            for p in getattr(self, attr).parameters()
            if p.requires_grad
        )
        n_heads = sum(
            p.numel()
            for h in self.heads.values()
            for p in h.parameters()
            if p.requires_grad
        )
        n_bb = sum(
            p.numel()
            for p in self.backbone.parameters()
            if p.requires_grad
        )
        n_total = n_tp + n_heads + n_bb
        print(
            f"[taskprompter] params — prompt: {n_tp / 1e6:.2f}M | "
            f"heads: {n_heads / 1e6:.2f}M | "
            f"backbone: {n_bb / 1e6:.2f}M | "
            f"total trainable: {n_total / 1e6:.2f}M"
        )

    @staticmethod
    def _get_cfg(obj, path: str, default=None):
        """安全遍历嵌套 dict/对象路径.

        兼容 Cfg (属性访问) 和普通 dict (下标访问).

        Args:
            obj: 起始对象 (Cfg / dict / 任意).
            path: 点分隔路径, 如 "tasks.cls.enabled".
            default: 路径不存在时返回的默认值.

        Returns:
            路径对应的值, 或 default.
        """
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = getattr(cur, key, default)
            if cur is default:
                break
        return cur
