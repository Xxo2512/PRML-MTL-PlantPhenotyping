"""计数: 叶尖计数 (Wheat Leaf Tip Counting), Pascal VOC XML.

每个 <object><name>tip</name><bndbox>...</bndbox></object> 表示一个叶尖, bbox
本身只作为点位置的承载, 取中心 ((xmin+xmax)/2, (ymin+ymax)/2) 为点 (x, y).

按契约, dataset 仅返回 raw points + count; density-map 由 cnt head 在训练时
用 targets['points'] 现算 (head 决定 stride / sigma, 见 models/heads/cnt_pet.py).

raw sample: {image_path, image_pil, points: np.ndarray [N,2] xy 像素 (原图), count: int}.
"""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .input_adapter import InputAdapter


def _parse_voc_xml(path: str) -> np.ndarray:
    """每个 object 的 bbox 中心 -> (N, 2) float32 in 原图像素坐标."""
    tree = ET.parse(path)
    root = tree.getroot()
    pts = []
    for obj in root.iter('object'):
        b = obj.find('bndbox')
        if b is None:
            continue
        x1 = float(b.findtext('xmin', '0'))
        y1 = float(b.findtext('ymin', '0'))
        x2 = float(b.findtext('xmax', '0'))
        y2 = float(b.findtext('ymax', '0'))
        pts.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


class WheatCountDataset(Dataset):
    """仅 train/val; 'test' 自动回退到 'val' (该集合无 test split)."""

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        train: Optional[bool] = None,
    ):
        if split == 'test':
            split = 'val'
        self.img_dir = os.path.join(root, 'images', split)
        self.ann_dir = os.path.join(root, 'annotations', split)
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform
        if self.transform is None and train is not None:
            self.transform = InputAdapter('cnt', train=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict[str, Any]:
        f = self.files[idx]
        stem = os.path.splitext(f)[0]
        path = os.path.join(self.img_dir, f)
        xml_path = os.path.join(self.ann_dir, stem + '.xml')
        points = _parse_voc_xml(xml_path) if os.path.isfile(xml_path) \
                 else np.zeros((0, 2), dtype=np.float32)
        sample = {
            'image_path': path,
            'image_pil':  Image.open(path),
            'points':     points,
            'count':      int(points.shape[0]),
        }
        return self.transform(sample) if self.transform is not None else sample
