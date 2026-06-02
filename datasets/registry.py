"""任务键、dataset/loader 工厂、单一入口。"""
from __future__ import annotations
from typing import Dict
import torch
from torch.utils.data import DataLoader

TASKS = ('seg', 'det', 'cnt', 'cls')   # 顺序固定


def build_dataset(task: str, split: str, cfg):
    """根据任务名分发到具体 Dataset。split: 'train' | 'val' | 'test'。"""
    if task == 'cls':
        from .wheat_cls import WheatGrowthStageDataset
        return WheatGrowthStageDataset(cfg.tasks.cls.data_root, split, train=(split == 'train'))
    if task == 'seg':
        from .wheat_seg import WheatOrganSegDataset
        return WheatOrganSegDataset(cfg.tasks.seg.data_root, split, train=(split == 'train'))
    if task == 'det':
        from .wheat_det import WheatHeadDetDataset
        return WheatHeadDetDataset(cfg.tasks.det.data_root, split, train=(split == 'train'))
    if task == 'cnt':
        from .wheat_cnt import WheatCountDataset
        return WheatCountDataset(cfg.tasks.cnt.data_root, split, train=(split == 'train'))
    raise ValueError(task)


def build_loader(task: str, split: str, cfg) -> DataLoader:
    ds = build_dataset(task, split, cfg)
    bs = cfg.data.batch_per_task
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == 'train'),
        num_workers=cfg.data.num_workers,
        collate_fn=ds.collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )


def build_all_loaders(cfg, split: str = 'train') -> Dict[str, DataLoader]:
    return {t: build_loader(t, split, cfg) for t in TASKS if cfg.tasks[t].enabled}
