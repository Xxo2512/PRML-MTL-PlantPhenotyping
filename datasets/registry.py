"""任务键 + dataset/loader 工厂 + 单一入口.

按 docs/api_contract.md §0 / §2 / §5:
  - dataset.__getitem__ 返回 raw (PIL + numpy/标量) sample
  - InputAdapter(task, train) 把 raw 变成 {image, targets, meta}
  - DataLoader 用 utils.collate.task_collate(batch, task) 合 batch
  - CrossDatasetSampler 在外层做跨任务混采
"""
from __future__ import annotations

from functools import partial
from typing import Dict

from torch.utils.data import DataLoader

from utils.collate import task_collate

from .input_adapter import InputAdapter
from .wheat_det import WheatDetectionDataset, WheatHeadDetDataset


TASKS = ("seg", "det", "cnt", "cls")


def _cfg_get(cfg, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        try:
            cur = cur[key]
        except (KeyError, TypeError):
            cur = getattr(cur, key, default)
        if cur is default:
            break
    return cur


def _build_adapter(task: str, split: str, cfg) -> InputAdapter:
    return InputAdapter(
        task=task,
        train=(split == "train"),
        input_size=int(_cfg_get(cfg, "model.input_size", 384)),
    )


def build_dataset(task: str, *args, split: str = "train", cfg=None):
    """根据任务名分发到具体 Dataset; 注入 InputAdapter 作为 transform.

    兼容历史调用风格: build_dataset(task, cfg, split='...') 与 build_dataset(task, split, cfg).
    """
    if cfg is None:
        if len(args) == 1:
            cfg = args[0]
        elif len(args) >= 2:
            split = args[0]
            cfg = args[1]
        else:
            raise TypeError("build_dataset requires cfg")

    adapter = _build_adapter(task, split, cfg)

    if task == "cls":
        from .wheat_cls import WheatGrowthStageDataset

        return WheatGrowthStageDataset(
            _cfg_get(cfg, "tasks.cls.data_root"), split, transform=adapter,
        )
    if task == "seg":
        from .wheat_seg import WheatOrganSegDataset

        return WheatOrganSegDataset(
            _cfg_get(cfg, "tasks.seg.data_root"), split, transform=adapter,
        )
    if task == "det":
        return WheatDetectionDataset(
            root=_cfg_get(cfg, "tasks.det.data_root"), split=split, transform=adapter,
        )
    if task == "cnt":
        from .wheat_cnt import WheatCountDataset

        return WheatCountDataset(
            _cfg_get(cfg, "tasks.cnt.data_root"), split, transform=adapter,
        )
    raise ValueError(task)


def build_loader(task: str, *args, split: str = "train", cfg=None) -> DataLoader:
    if cfg is None:
        if len(args) == 1:
            cfg = args[0]
        elif len(args) >= 2:
            split = args[0]
            cfg = args[1]
        else:
            raise TypeError("build_loader requires cfg")

    dataset = build_dataset(task, split, cfg)
    return DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "data.batch_per_task", 8)),
        shuffle=(split == "train"),
        num_workers=int(_cfg_get(cfg, "data.num_workers", 0)),
        collate_fn=partial(task_collate, task=task),
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_all_loaders(cfg, split: str = "train") -> Dict[str, DataLoader]:
    return {
        task: build_loader(task, split, cfg)
        for task in TASKS
        if bool(_cfg_get(cfg, f"tasks.{task}.enabled", True))
    }


__all__ = [
    "TASKS",
    "WheatDetectionDataset",
    "WheatHeadDetDataset",
    "build_dataset",
    "build_loader",
    "build_all_loaders",
]
