from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .input_adapter import InputAdapter


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class WheatDetectionDataset(Dataset):
    """Wheat head detection dataset stored as images plus YOLO txt labels."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[InputAdapter] = None,
        train: Optional[bool] = None,
    ):
        self.root = Path(root)
        if self.root.name != "detect_dataset" and (self.root / "detect_dataset").exists():
            self.root = self.root / "detect_dataset"
        self.split = split
        self.transform = transform
        if self.transform is None and train is not None:
            self.transform = InputAdapter("det", train=train)

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image split directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label split directory not found: {self.label_dir}")

        self.images = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self._read_yolo_label(image_path.stem, image.size)
        sample = {
            "image_path": str(image_path),
            "image_pil": image,
            "boxes": boxes,
            "labels": labels,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _read_yolo_label(self, stem: str, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        label_path = self.label_dir / f"{stem}.txt"
        width, height = image_size
        rows: List[List[float]] = []
        labels: List[int] = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = parts
                    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                    x1 = (cx - bw / 2.0) * width
                    y1 = (cy - bh / 2.0) * height
                    x2 = (cx + bw / 2.0) * width
                    y2 = (cy + bh / 2.0) * height
                    rows.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        boxes = np.asarray(rows, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.asarray(labels, dtype=np.int64)
        return boxes, labels_arr

    @staticmethod
    def collate_fn(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "targets": {
                "boxes": [item["targets"]["boxes"] for item in batch],
                "labels": [item["targets"]["labels"] for item in batch],
            },
            "meta": [item.get("meta", {}) for item in batch],
        }


class WheatHeadDetDataset(WheatDetectionDataset):
    """Main-line compatible name for the detection dataset."""

    def __init__(self, root: str | Path, split: str, train: bool):
        super().__init__(root=root, split=split, transform=InputAdapter("det", train=train))
