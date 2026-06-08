"""分割: 4 类 (含背景) 小麦器官分割.

raw sample: {image_path: str, image_pil: PIL.Image, mask: np.ndarray [H,W] uint8}.
带 transform=InputAdapter('seg', ...) 时输出按契约 {image, targets:{mask}, meta}.

mask 像素值 ∈ {0: Background, 1: Head, 2: Stem, 3: Leaf}.
"""
from __future__ import annotations
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .input_adapter import InputAdapter


class WheatOrganSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        train: Optional[bool] = None,
    ):
        self.img_dir  = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'class_id')
        self.files = sorted([
            f for f in os.listdir(self.img_dir) if f.lower().endswith('.png')
        ])
        self.transform = transform
        if self.transform is None and train is not None:
            self.transform = InputAdapter('seg', train=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, Any]:
        f = self.files[idx]
        img_path  = os.path.join(self.img_dir, f)
        mask_path = os.path.join(self.mask_dir, f)
        sample = {
            'image_path': img_path,
            'image_pil':  Image.open(img_path),
            'mask':       np.array(Image.open(mask_path), dtype=np.uint8),
        }
        return self.transform(sample) if self.transform is not None else sample
