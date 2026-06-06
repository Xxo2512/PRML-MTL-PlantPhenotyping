from __future__ import annotations

from typing import Dict, List

import torch


def task_collate(batch: List[Dict], task: str) -> Dict:
    """Collate one task batch while keeping variable-length targets as lists."""

    images = torch.stack([item["image"] for item in batch], dim=0)
    metas = [item.get("meta", {"task": task}) for item in batch]

    if task == "det":
        targets = {
            "boxes": [item["targets"]["boxes"] for item in batch],
            "labels": [item["targets"]["labels"] for item in batch],
        }
    elif task == "seg":
        targets = {"mask": torch.stack([item["targets"]["mask"] for item in batch], dim=0)}
    elif task == "cnt":
        targets = {
            "points": [item["targets"]["points"] for item in batch],
            "count": torch.stack([item["targets"]["count"] for item in batch], dim=0),
        }
    elif task == "cls":
        targets = {"label": torch.stack([item["targets"]["label"] for item in batch], dim=0)}
    else:
        raise ValueError(f"Unknown task: {task}")

    return {"image": images, "targets": targets, "meta": metas}
