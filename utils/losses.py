"""Loss aggregator: uniform / dwa / uncertainty (Kendall'18)。

uniform:
    L = (1/|T|) Σ_t L_t

uncertainty:
    L = Σ_t  0.5 * exp(-s_t) * L_t  +  0.5 * s_t,  其中 s_t = log σ_t² 可学
    (Kendall et al., CVPR'18)

dwa (W14 之后再实现):
    使用最近若干 step 的 loss 比率近似动态权重。
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn


class LossAggregator(nn.Module):
    def __init__(self, mode: str = 'uniform', tasks: tuple = ('seg', 'det', 'cnt', 'cls')):
        super().__init__()
        assert mode in ('uniform', 'uncertainty')      # dwa 后续再加
        self.mode = mode
        self.tasks = tasks
        if mode == 'uncertainty':
            # 每个 task 一个可学的 log σ²; 全 0 初始
            self.log_var = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(())) for t in tasks}
            )

    def forward(self, per_task_loss: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == 'uniform':
            losses = list(per_task_loss.values())
            return torch.stack(losses).mean()
        # uncertainty
        total = 0.0
        for t, L in per_task_loss.items():
            s = self.log_var[t]
            total = total + 0.5 * torch.exp(-s) * L + 0.5 * s
        return total
