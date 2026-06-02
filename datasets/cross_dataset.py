"""跨数据集采样器 (MTDNN-style)。

数学:
  P(t) ∝ |D_t|^α,  α ∈ [0, 1]
  α=0  : 等概率 (≈ round-robin)
  α=1  : 严格按数据量
  α=0.3 - 0.5 : 平衡大/小集 (经验值)

每次产出 {task: batch}, 长度 = 1 (RR/PS) 或 = #tasks (HM)。
"""
from __future__ import annotations
import random
from typing import Dict, Iterator
from torch.utils.data import DataLoader


class CrossDatasetSampler:
    def __init__(self, loaders: Dict[str, DataLoader], mode: str = 'ps',
                 alpha: float = 0.5, length: int | None = None):
        assert mode in ('rr', 'ps', 'hm')
        self.loaders = loaders
        self.iters = {t: iter(L) for t, L in loaders.items()}
        self.tasks = list(loaders.keys())
        self.mode = mode
        self.alpha = alpha
        sizes = {t: len(L) for t, L in loaders.items()}
        self.sizes = sizes
        # PS 概率
        weights = {t: (sizes[t] ** alpha) for t in self.tasks}
        s = sum(weights.values())
        self.probs = [weights[t] / s for t in self.tasks]
        # 默认一个 epoch 的步数 = max(sizes)
        self.length = length if length is not None else max(sizes.values())
        self._step = 0
        self._rr_cursor = 0

    def __iter__(self) -> Iterator[Dict[str, dict]]:
        self._step = 0
        return self

    def __next__(self) -> Dict[str, dict]:
        if self._step >= self.length:
            raise StopIteration
        self._step += 1
        if self.mode == 'hm':
            return {t: self._take(t) for t in self.tasks}
        if self.mode == 'rr':
            t = self.tasks[self._rr_cursor % len(self.tasks)]
            self._rr_cursor += 1
        else:  # 'ps'
            t = random.choices(self.tasks, weights=self.probs, k=1)[0]
        return {t: self._take(t)}

    def __len__(self):
        return self.length

    def _take(self, t: str):
        try:
            return next(self.iters[t])
        except StopIteration:
            self.iters[t] = iter(self.loaders[t])
            return next(self.iters[t])
