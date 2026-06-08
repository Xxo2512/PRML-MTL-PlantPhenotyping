"""分类: 6 类小麦生育期 (ImageFolder 风格).

raw sample (__getitem__ 不带 transform 时):
  {image_path: str, image_pil: PIL.Image, label: int}

带 transform=InputAdapter('cls', train=...) 时, 输出按契约:
  {image: [3,384,384], targets: {label}, meta: {...}}
"""
from __future__ import annotations
import os
from typing import Any, Callable, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from .input_adapter import InputAdapter

CLASSES = ['1_Tillering', '2_Jointing', '3_BH', '4_Flowering', '5_Filling', '6_Ripening']
NAME2IDX = {c: i for i, c in enumerate(CLASSES)}


class WheatGrowthStageDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        train: Optional[bool] = None,
    ):
        split_dir = os.path.join(root, split)
        self.samples: list[tuple[str, int]] = []
        for cname in CLASSES:
            cdir = os.path.join(split_dir, cname)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cdir, f), NAME2IDX[cname]))
        self.transform = transform
        if self.transform is None and train is not None:
            self.transform = InputAdapter('cls', train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        path, label = self.samples[idx]
        sample = {
            'image_path': path,
            'image_pil':  Image.open(path),
            'label':      int(label),
        }
        return self.transform(sample) if self.transform is not None else sample
