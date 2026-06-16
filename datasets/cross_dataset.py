"""跨数据集采样器 (MTDNN-style)。

老师 W14 反馈: 外层采样需构造成"所有任务 batch 的任务序列, shuffle 一遍",
一个 epoch 的长度 = 各任务 batch 数之和 (而非 max)。

数学:
  pure proportional (MTDNN α=1):
      seq = [t1]*|D_t1| + [t2]*|D_t2| + ...,  random.shuffle(seq)
      epoch_len = Σ |D_t|
      → 每个 task 的每个 batch 恰好被消费 1 次

  α-balanced (本实现, α ∈ [0, 1]):
      share(t)  = |D_t|^α / Σ_τ |D_τ|^α          # task t 在一个 epoch 中的步数占比
      count(t)  = round(share(t) × Σ_τ |D_τ|)    # task t 在一个 epoch 中出现的次数
      seq       = concat_t [t]*count(t), random.shuffle(seq)
      epoch_len = Σ count(t) ≈ Σ |D_t|

      α=1   ⇒ 严格 proportional, 每 batch 恰好一次 (MTDNN 原版)
      α=0.5 ⇒ 小任务被复用 (其 batch 被采样 count(t)/|D_t| 次), 大任务被欠采 (随机抽 count(t) 个 batch),
              既保留小任务领域信号又防止大任务 (cls=30527 batches) 一家独大
      α=0   ⇒ 每个 task 步数均等 (≈ round-robin, 但顺序仍 shuffle)

模式:
  - 'ps' : α-balanced (默认 α=0.5)
  - 'rr' : 严格轮转 (epoch_len = Σ |D_t|, 每步换下一个 task)
  - 'hm' : homogeneous, 每步同时产 4 个 task 的 batch (epoch_len = max |D_t|)

每次 __next__ 产出 {task: batch}, 长度 = 1 (rr/ps) 或 = #tasks (hm)。
迭代器耗尽时自动 reset, 保证 task 序列对齐。
"""
from __future__ import annotations
import random
from typing import Dict, Iterator, List, Optional
from torch.utils.data import DataLoader


class CrossDatasetSampler:
    def __init__(self, loaders: Dict[str, DataLoader], mode: str = 'ps',
                 alpha: float = 0.5, length: Optional[int] = None,
                 seed: Optional[int] = None):
        assert mode in ('rr', 'ps', 'hm')
        self.loaders = loaders
        self.tasks: List[str] = list(loaders.keys())
        self.mode = mode
        self.alpha = alpha
        self.sizes: Dict[str, int] = {t: len(L) for t, L in loaders.items()}
        self._rng = random.Random(seed)

        if mode == 'hm':
            self.epoch_seq: List[str] = []
            self.length = length if length is not None else max(self.sizes.values())
        else:
            total = sum(self.sizes.values())
            if mode == 'rr':
                # 严格轮转: 长度 = sum(sizes), 序列 = [t1, t2, ..., tN, t1, t2, ...] 重复至填满
                self.counts: Dict[str, int] = {t: 0 for t in self.tasks}
                seq: List[str] = []
                cursor = 0
                target = length if length is not None else total
                while len(seq) < target:
                    t = self.tasks[cursor % len(self.tasks)]
                    seq.append(t)
                    self.counts[t] += 1
                    cursor += 1
                self.epoch_seq = seq
            else:  # 'ps' — α-balanced shuffled
                weights = {t: (max(self.sizes[t], 1) ** alpha) for t in self.tasks}
                wsum = sum(weights.values())
                # share(t) * total, 向下取整, 再用最大余数法分配剩余
                raw = {t: weights[t] / wsum * total for t in self.tasks}
                counts = {t: int(raw[t]) for t in self.tasks}
                remainder = total - sum(counts.values())
                # 按 raw 的小数部分降序补齐
                frac_order = sorted(self.tasks, key=lambda t: raw[t] - counts[t], reverse=True)
                for i in range(remainder):
                    counts[frac_order[i % len(frac_order)]] += 1
                self.counts = counts
                seq: List[str] = []
                for t in self.tasks:
                    seq.extend([t] * counts[t])
                self._rng.shuffle(seq)
                self.epoch_seq = seq
            self.length = length if length is not None else len(self.epoch_seq)
            # 若用户传入 length 小于 epoch_seq, 截断 (smoke 用)
            if self.length < len(self.epoch_seq):
                self.epoch_seq = self.epoch_seq[:self.length]

        self.iters = {t: iter(L) for t, L in loaders.items()}
        self._step = 0

    def __iter__(self) -> Iterator[Dict[str, dict]]:
        self._step = 0
        # 每次进入新 epoch 时, ps 重新 shuffle 序列 (rr/hm 保持原序列)
        if self.mode == 'ps':
            self._rng.shuffle(self.epoch_seq)
        return self

    def __next__(self) -> Dict[str, dict]:
        if self._step >= self.length:
            raise StopIteration
        if self.mode == 'hm':
            self._step += 1
            return {t: self._take(t) for t in self.tasks}
        t = self.epoch_seq[self._step]
        self._step += 1
        return {t: self._take(t)}

    def __len__(self) -> int:
        return self.length

    def task_counts(self) -> Dict[str, int]:
        """返回当前 epoch 里每个 task 出现的次数 (用于日志/sanity check)。"""
        if self.mode == 'hm':
            return {t: self.length for t in self.tasks}
        return dict(getattr(self, 'counts', {t: self.epoch_seq.count(t) for t in self.tasks}))

    def _take(self, t: str):
        try:
            return next(self.iters[t])
        except StopIteration:
            self.iters[t] = iter(self.loaders[t])
            return next(self.iters[t])
