"""所有 task head 的抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class BaseTaskHead(nn.Module, ABC):
    task: str
    in_channels: List[Optional[int]]   # 与 backbone out_channels 对齐, None 表示该 stage 不消费

    @abstractmethod
    def forward(self, feats: Dict[str, torch.Tensor],
                targets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        train  : {'loss': scalar tensor, 'loss_items': dict[str,float], 'pred': any}
        eval   : {'pred': any, 'metric': dict[str,float]}
        """
