from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from utils.collate import task_collate

from .input_adapter import InputAdapter
from .wheat_det import WheatDetectionDataset


TASKS = ("seg", "det", "cnt", "cls")


def build_dataset(task: str, cfg: Dict, split: str = "train"):
    task_cfg = cfg["tasks"][task]
    root = Path(task_cfg["data_root"])
    adapter = InputAdapter(task=task, train=(split == "train"), input_size=cfg["model"]["input_size"])

    if task == "det":
        return WheatDetectionDataset(root=root, split=split, transform=adapter)
    raise NotImplementedError(
        f"Dataset for task '{task}' is owned by another member and is not implemented yet."
    )


def build_loader(task: str, cfg: Dict, split: str = "train") -> DataLoader:
    dataset = build_dataset(task, cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_per_task"],
        shuffle=(split == "train"),
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=partial(task_collate, task=task),
    )
