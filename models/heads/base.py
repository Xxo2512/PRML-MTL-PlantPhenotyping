from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch.nn as nn


class BaseTaskHead(nn.Module, ABC):
    task: str
    in_channels: List[Optional[int]]

    @abstractmethod
    def forward(self, feats: Dict, targets=None) -> Dict[str, Any]:
        raise NotImplementedError
