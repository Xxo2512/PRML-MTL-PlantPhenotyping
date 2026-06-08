"""PGT — Prompt-Guided Transformer for multi-task learning.

参考: Lu et al., "Prompt Guided Transformer for Multi-task Dense Prediction",
      IEEE TMM, 2024.

核心思想:
  在共享 backbone 的每个 stage 注入可学习 prompt token,
  通过 prompt-guided cross-attention 让 prompt 从图像特征中
  提取任务相关信息, 然后用 FiLM 风格仿射变换调制特征。

架构:
  1. Shared Swin backbone 提取 4 级多尺度特征 {s1, s2, s3, s4}
  2. 每个 task × stage 持有 n_prompt 个可学习 token
  3. Prompt tokens (query) 与 投影后的特征 (key/value) 做 cross-attention
  4. 聚合 prompt 输出 → 生成 per-channel γ, β → 调制原特征
  5. 调制后特征送入各自 task head

与项目框架的对接:
  - 继承 MTLModel, 不改父类签名
  - forward 接受 batch dict (支持 len=1 的 RR/PS 及 len=4 的 HM)
  - 支持 compute_loss / compute_metrics / get_optimizer_params
  - backbone 冻结时仅训练 prompt + projection + head 参数
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MTLModel


# ══════════════════════════════════════════════════════════════════════
# Prompt-guided Cross-Attention 模块
# ══════════════════════════════════════════════════════════════════════


class PromptCrossAttention(nn.Module):
    """Prompt 作为 query, 图像特征作为 key/value 的交叉注意力.

    数学:
      Q = W_q · prompts          ∈ R^{B × n_prompts × d}
      K = W_k · feat_flat        ∈ R^{B × HW × d}
      V = W_v · feat_flat        ∈ R^{B × HW × d}
      A = softmax(QK^T / √d)     ∈ R^{B × n_prompts × HW}
      O = W_o · (A · V)          ∈ R^{B × n_prompts × d}

    聚合所有 prompt 输出 → mean pooling → FiLM 调制 (γ, β):
      γ = W_γ · prompt_feat,  β = W_β · prompt_feat
      feat' = feat ⊙ (1 + γ) + β

    Args:
        dim: 特征/投影维度.
        n_prompts: prompt token 数量.
        n_heads: 多头注意力的头数.
    """

    def __init__(self, dim: int, n_prompts: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} 必须能被 n_heads {n_heads} 整除"

        self.n_prompts = n_prompts
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(dim, dim)       # prompt → query
        self.k_proj = nn.Linear(dim, dim)       # feature → key
        self.v_proj = nn.Linear(dim, dim)       # feature → value
        self.out_proj = nn.Linear(dim, dim)

        # FiLM 调制生成器: prompt 聚合特征 → per-channel (γ, β)
        self.modulation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化投影层, 零初始化调制层的最后一层 bias (恒等起始)."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        # 调制层最后一层: 零初始 → 初始时不改变特征 (γ=0, β=0)
        last_linear = self.modulation[-1]
        nn.init.xavier_uniform_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, prompts: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            prompts: [B, n_prompts, dim] — 可学习 prompt token (已扩展 batch).
            feat:    [B, C, H, W] — 投影后的图像特征图 (C == dim).

        Returns:
            调制后的特征图 [B, C, H, W].
        """
        B, C, H, W = feat.shape

        # 空间展平: [B, C, H, W] → [B, HW, C]
        feat_flat = feat.flatten(2).transpose(1, 2)           # [B, HW, C]

        # 投影
        Q = self.q_proj(prompts)                               # [B, n_prompts, C]
        K = self.k_proj(feat_flat)                             # [B, HW, C]
        V = self.v_proj(feat_flat)                             # [B, HW, C]

        # 多头 reshape: [B, seq, C] → [B, n_heads, seq, head_dim]
        Q = Q.view(B, self.n_prompts, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale        # [B, n_heads, n_prompts, HW]
        attn = attn.softmax(dim=-1)

        # 加权聚合
        attended = attn @ V                                    # [B, n_heads, n_prompts, head_dim]
        attended = attended.transpose(1, 2).contiguous()      # [B, n_prompts, n_heads, head_dim]
        attended = attended.view(B, self.n_prompts, C)        # [B, n_prompts, C]

        # 输出投影 + 聚合所有 prompt
        attended = self.out_proj(attended)                     # [B, n_prompts, C]
        prompt_feat = attended.mean(dim=1)                     # [B, C]

        # 生成 per-channel 调制参数
        gamma_beta = self.modulation(prompt_feat)              # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)              # [B, C], [B, C]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        return feat * (1.0 + gamma) + beta


# ══════════════════════════════════════════════════════════════════════
# Feature Projection 模块
# ══════════════════════════════════════════════════════════════════════


class FeatureProjection(nn.Module):
    """将 backbone 特征投影到统一 prompt 维度, 含可选的 dim 对齐.

    不同 stage 输出通道不同 (96, 192, 384, 768),
    投影到统一 dim 后便于 prompt cross-attention 统一处理.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════
# PGT MTL 模型主体
# ══════════════════════════════════════════════════════════════════════


class PGTModel(MTLModel):
    """Prompt-Guided Transformer 多任务模型.

    设计要点:
      - 每个启用的 task 持有 per-stage 的可学习 prompt token
      - Prompt 通过 cross-attention 从图像特征中提取任务相关信息
      - 提取的信息以 FiLM 调制方式反馈到 backbone 特征
      - Backbone 默认冻结, 仅训练 prompt + projection + cross-attn + head 参数
      - 完全兼容 train.py 与 evaluate.py 的接口

    Config (configs/method/pgt.yaml):
      model:
        freeze_backbone: true        # 冻结 backbone
        pgt:
          n_prompt: 8                # prompt token 数量
          use_cross_attn: true       # 是否启用 cross-attention 调制
          prompt_dim: 256            # 统一投影维度
          n_heads: 8                 # 交叉注意力头数
    """

    def __init__(
        self,
        cfg=None,
        backbone: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None,
    ):
        # ---- 父类: 构建 backbone + heads ----
        super().__init__(cfg=cfg, backbone=backbone, heads=heads)

        # ---- 读取 PGT 专属配置 ----
        model_cfg = self._get_cfg(cfg, "model", {})
        pgt_cfg = self._get_cfg(model_cfg, "pgt", {})

        self.n_prompt = int(self._get_cfg(pgt_cfg, "n_prompt", 8))
        self.use_cross_attn = bool(self._get_cfg(pgt_cfg, "use_cross_attn", True))
        prompt_dim = int(self._get_cfg(pgt_cfg, "prompt_dim", 256))
        n_heads = int(self._get_cfg(pgt_cfg, "n_heads", 8))

        # ---- 确定启用的任务列表 ----
        self.tasks = tuple(
            task
            for task in ("seg", "det", "cnt", "cls")
            if self._get_cfg(cfg, f"tasks.{task}.enabled", True)
        )

        # ---- Backbone 输出通道 ----
        out_channels: List[int] = getattr(
            self.backbone, "out_channels", [96, 192, 384, 768]
        )
        self.out_channels = out_channels
        stage_keys = [f"s{i + 1}" for i in range(len(out_channels))]

        # ---- 1) Prompt tokens: 每个 task × stage 持有 n_prompt 个 token ----
        self.prompt_tokens = nn.ParameterDict()
        for task in self.tasks:
            # [1, n_prompt, prompt_dim] — forward 时 expand 到 batch
            self.prompt_tokens[task] = nn.Parameter(
                torch.randn(1, self.n_prompt, prompt_dim) * 0.02
            )

        # ---- 2) 特征投影: backbone channel → prompt_dim ----
        self.feat_projections = nn.ModuleDict()
        for task in self.tasks:
            self.feat_projections[task] = nn.ModuleDict({
                stage_key: FeatureProjection(ch, prompt_dim)
                for stage_key, ch in zip(stage_keys, out_channels)
            })

        # ---- 3) Prompt-guided cross-attention (可选) ----
        if self.use_cross_attn:
            self.cross_attn = nn.ModuleDict()
            for task in self.tasks:
                self.cross_attn[task] = nn.ModuleDict({
                    stage_key: PromptCrossAttention(prompt_dim, self.n_prompt, n_heads)
                    for stage_key in stage_keys
                })

        # ---- 4) 输出投影: prompt_dim → 原始 backbone channel ----
        #       用于残差连接: feat + project_back(modulated_feat)
        self.out_projections = nn.ModuleDict()
        for task in self.tasks:
            self.out_projections[task] = nn.ModuleDict({
                stage_key: nn.Conv2d(prompt_dim, ch, kernel_size=1)
                for stage_key, ch in zip(stage_keys, out_channels)
            })
            # 零初始化输出投影, 使初始状态恒等
            for conv in self.out_projections[task].values():
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

        # ---- 冻结 backbone (按配置) ----
        if self._get_cfg(model_cfg, "freeze_backbone", False):
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """多任务前向传播.

        流程 (每个 task 独立):
          1. Backbone 提取多尺度特征
          2. PGT 调制: prompt cross-attn → FiLM 调制特征
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

            # Step 2: PGT 调制 (如果启用)
            if self.use_cross_attn:
                feats = self._apply_pgt(task, feats)

            # Step 3: Task head
            outputs[task] = self.heads[task](feats, task_batch.get("targets"))

        return outputs

    # ------------------------------------------------------------------
    # PGT 核心: prompt-guided 特征调制
    # ------------------------------------------------------------------

    def _apply_pgt(
        self, task: str, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """对 backbone 输出的每个 stage 特征施加 PGT 调制.

        Pipeline:
          1. 扩展 prompt token 到当前 batch size
          2. 特征投影到 prompt_dim
          3. Cross-attention: prompt (query) × feature (key/value)
          4. 输出投影回原始通道, 残差连接

        Args:
            task:  任务名.
            feats: {'s1': [B,96,96,96], 's2': [B,192,48,48],
                    's3': [B,384,24,24], 's4': [B,768,12,12]}

        Returns:
            调制后的特征 dict (相同 keys 和 shapes).
        """
        B = next(iter(feats.values())).shape[0]

        # 扩展 prompt: [1, n_prompt, prompt_dim] → [B, n_prompt, prompt_dim]
        prompts = self.prompt_tokens[task].expand(B, -1, -1)

        modulated: Dict[str, torch.Tensor] = {}
        for stage_key in feats:
            feat = feats[stage_key]

            # 投影到 prompt_dim
            feat_proj = self.feat_projections[task][stage_key](feat)

            # Cross-attention 调制
            feat_mod = self.cross_attn[task][stage_key](prompts, feat_proj)

            # 投影回原始通道
            feat_out = self.out_projections[task][stage_key](feat_mod)

            # 残差连接: 保留 backbone 原始特征
            modulated[stage_key] = feat + feat_out

        return modulated

    # ------------------------------------------------------------------
    # Loss & Metrics
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
        """计算分类及其他任务的评测指标.

        Args:
            outputs: model(batch) 的返回值.
            batch:   原始输入 batch (含 targets).

        Returns:
            Flat dict: {'cls/acc': 0.85, 'cls/mAP': 0.82, 'cls/BA': 0.80, ...}
        """
        from utils.metrics import compute_cls_metric, compute_seg_metric, compute_det_metric, compute_cnt_metric

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
                    all_metrics.update(compute_det_metric(preds, boxes_gt, labels_gt))

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
          - PGT 专属参数 (prompt, projection, cross-attn): lr
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

        # Group 2: PGT 专属参数
        # 收集 prompt_tokens, feat_projections, cross_attn, out_projections 中的参数
        pgt_param_ids = set()
        for attr_name in ["prompt_tokens", "feat_projections", "cross_attn",
                          "out_projections"]:
            if hasattr(self, attr_name):
                module = getattr(self, attr_name)
                for p in module.parameters():
                    if p.requires_grad:
                        pgt_param_ids.add(id(p))
        if pgt_param_ids:
            param_groups.append({
                "params": [p for p in self.parameters()
                           if id(p) in pgt_param_ids and p.requires_grad],
                "lr": lr,
                "weight_decay": weight_decay,
            })

        # Group 3: Head 参数
        head_params = []
        for head in self.heads.values():
            for p in head.parameters():
                if p.requires_grad and id(p) not in pgt_param_ids:
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
            # 冻结的 backbone 始终 eval (避免 BN/Dropout 更新)
            model_cfg = self._get_cfg(self.cfg, "model", {})
            if self._get_cfg(model_cfg, "freeze_backbone", False):
                self.backbone.eval()
        return self

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
