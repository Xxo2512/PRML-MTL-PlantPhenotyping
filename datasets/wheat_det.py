"""检测: 麦穗 (单类), YOLO 标注。"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .input_adapter import adapt_det

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def _yolo_to_xyxy_norm(arr: np.ndarray) -> np.ndarray:
    """YOLO (cls cx cy w h, normalized) -> (x1 y1 x2 y2, normalized)."""
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


class WheatHeadDetDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool):
        self.img_dir = os.path.join(root, 'images', split)
        self.lbl_dir = os.path.join(root, 'labels', split)
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(IMG_EXTS)])
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        stem = os.path.splitext(f)[0]
        img = Image.open(os.path.join(self.img_dir, f))
        lbl_path = os.path.join(self.lbl_dir, stem + '.txt')
        if os.path.isfile(lbl_path) and os.path.getsize(lbl_path) > 0:
            arr = np.loadtxt(lbl_path, dtype=np.float32, ndmin=2)
        else:
            arr = np.zeros((0, 5), dtype=np.float32)
        boxes = _yolo_to_xyxy_norm(arr)
        x, boxes = adapt_det(img, boxes, self.train)
        return {'image': x, 'boxes': torch.from_numpy(boxes)}

    @staticmethod
    def collate_fn(batch):
        # 框数量不一, targets 用 list-of-tensor
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'targets': {'boxes': [b['boxes'] for b in batch]},
        }
