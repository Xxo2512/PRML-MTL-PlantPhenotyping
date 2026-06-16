"""MTLoRA: Low-Rank Adaptation for Efficient Multi-Task Learning (CVPR 2024)。

数学:
  对于 backbone 中的某个 Linear 层 y = W₀ x + b：
    y = W₀ x + b
        + (α/r) · B_TA · A_TA · x          (Task-Agnostic, 全 task 共享)
        + (α/r) · B_t  · A_t  · x          (Task-Specific, t 为当前 task)
  其中 W₀ 冻结, A ∈ R^{r×in}, B ∈ R^{out×r}, A 用 Kaiming 初始化, B 初始化为 0
  → 训练初期等价于原 backbone, 仅 LoRA 增量贡献逐步学习。

实现:
  - 把 Swin 中所有 qkv/proj/fc1/fc2 的 nn.Linear 包成 LoRALinear。
  - LoRALinear 维护 current_task; 每次 backbone forward 前由 MTLoRAModel 统一 set。
  - backbone (timm) 完全冻结; 仅 LoRA 参数与 task heads 训练。
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import math

import torch
import torch.nn as nn

from .base import MTLModel


# ----------------------------------------------------------------------
# LoRA 单层
# ----------------------------------------------------------------------
class LoRALinear(nn.Module):
    """一个 nn.Linear 包装: y = base(x) + Σ scale · B A x。

    scale = alpha / rank
    base 完全冻结, A/B 是低秩学习参数。
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        tasks: Tuple[str, ...] = ('seg', 'det', 'cnt', 'cls'),
        use_task_agnostic: bool = True,
        use_task_specific: bool = True,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scale = alpha / rank if rank > 0 else 1.0
        in_f = base.in_features
        out_f = base.out_features

        # TA-LoRA (task-agnostic)
        if use_task_agnostic and rank > 0:
            self.ta_A = nn.Parameter(torch.empty(rank, in_f))
            self.ta_B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.ta_A, a=math.sqrt(5))
        else:
            self.register_parameter('ta_A', None)
            self.register_parameter('ta_B', None)

        # TS-LoRA (task-specific): 每 task 一份
        if use_task_specific and rank > 0:
            self.ts_A = nn.ParameterDict()
            self.ts_B = nn.ParameterDict()
            for t in tasks:
                a = nn.Parameter(torch.empty(rank, in_f))
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                b = nn.Parameter(torch.zeros(out_f, rank))
                self.ts_A[t] = a
                self.ts_B[t] = b
        else:
            self.ts_A = None
            self.ts_B = None

        self.current_task: Optional[str] = None

    def set_task(self, task: Optional[str]):
        self.current_task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # x 形状任意, 末维 = in_features
        if self.ta_A is not None:
            y = y + self.scale * torch.matmul(torch.matmul(x, self.ta_A.t()), self.ta_B.t())
        if (
            self.ts_A is not None
            and self.current_task is not None
            and self.current_task in self.ts_A
        ):
            a = self.ts_A[self.current_task]
            b = self.ts_B[self.current_task]
            y = y + self.scale * torch.matmul(torch.matmul(x, a.t()), b.t())
        return y


# ----------------------------------------------------------------------
# 注入工具: 把 backbone 中匹配名字的 Linear 替换为 LoRALinear
# ----------------------------------------------------------------------
def _inject_lora_layers(
    root: nn.Module,
    target_basenames: Iterable[str],
    rank: int,
    alpha: float,
    tasks: Tuple[str, ...],
    use_ta: bool,
    use_ts: bool,
) -> List[LoRALinear]:
    target_set = set(target_basenames)
    injected: List[LoRALinear] = []
    # 收集要替换的 (parent_module, attr_name) 对, 避免在迭代时改 module 树
    to_replace: List[Tuple[nn.Module, str, nn.Linear]] = []
    for name, mod in root.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        base_name = name.rsplit('.', 1)[-1]
        if base_name not in target_set:
            continue
        parent_name, _, attr = name.rpartition('.')
        parent = root.get_submodule(parent_name) if parent_name else root
        to_replace.append((parent, attr, mod))

    for parent, attr, base in to_replace:
        lora = LoRALinear(
            base, rank=rank, alpha=alpha, tasks=tasks,
            use_task_agnostic=use_ta, use_task_specific=use_ts,
        )
        setattr(parent, attr, lora)
        injected.append(lora)
    return injected


# ----------------------------------------------------------------------
# MTLoRAModel
# ----------------------------------------------------------------------
class MTLoRAModel(MTLModel):
    """共享 Swin backbone + LoRA 注入 + 4 个 task head。

    - backbone 完全冻结, 仅 LoRA + heads + (可选) loss σ 训练。
    - forward(batch) 中, 进入 backbone 前给所有 LoRA 层 set_task(t)。
    """

    def __init__(self, cfg):
        # 让父类按 cfg 建 backbone + heads, 并冻结 backbone (cfg.model.freeze_backbone=True)
        super().__init__(cfg=cfg)
        m = cfg.model.mtlora
        rank = int(getattr(m, 'rank', 16))
        alpha = float(getattr(m, 'alpha', 16.0))
        targets = list(getattr(m, 'target_modules', ['qkv', 'proj', 'fc1', 'fc2']))
        use_ta = bool(getattr(m, 'use_task_agnostic', True))
        use_ts = bool(getattr(m, 'use_task_specific', True))
        tasks = tuple(self.heads.keys())

        # 注入到 backbone (timm SwinTransformer 在 self.backbone.model)
        timm_model = self.backbone.model
        self._lora_layers: List[LoRALinear] = _inject_lora_layers(
            timm_model, targets, rank, alpha, tasks, use_ta, use_ts,
        )
        n_lora = sum(p.numel() for layer in self._lora_layers for p in layer.parameters() if p.requires_grad)
        print(f'[mtlora] injected {len(self._lora_layers)} LoRA layers, '
              f'rank={rank} alpha={alpha} TA={use_ta} TS={use_ts}; '
              f'LoRA params={n_lora/1e6:.2f}M')

    def _set_task(self, task: Optional[str]):
        for layer in self._lora_layers:
            layer.set_task(task)

    def forward(self, batch: Dict[str, Dict]) -> Dict[str, Dict]:
        outputs: Dict[str, Dict] = {}
        for task, task_batch in batch.items():
            if task not in self.heads:
                continue
            self._set_task(task)
            feats = self.backbone(task_batch['image'], task=task)
            outputs[task] = self.heads[task](feats, task_batch.get('targets'))
        # 防止 LoRA 状态泄漏到下次调用
        self._set_task(None)
        return outputs
