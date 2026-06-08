"""计数: 叶尖计数 (Wheat Leaf Tip Counting)。

数据集格式: Pascal VOC XML, 每张图一个 .xml,
  <object><name>tip</name><bndbox><xmin/><ymin/><xmax/><ymax/></bndbox></object>

"""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .input_adapter import adapt_cnt, INPUT_SIZE

DENSITY_STRIDE = 8                       # density map 在 stride=8 的特征图上
DENSITY_SIZE   = INPUT_SIZE // DENSITY_STRIDE   # 48
DENSITY_SIGMA  = 2.0


def _parse_voc_xml(path: str) -> np.ndarray:
    """返回 (N, 2) float32: 每个 object bbox 的中心 (x, y) in 原图像素坐标。"""
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


def _gaussian_density(points_xy: np.ndarray) -> np.ndarray:
    """points_xy: (N, 2) in pixel coords of INPUT_SIZE; 输出 (DENSITY_SIZE, DENSITY_SIZE) float32。"""
    H = W = DENSITY_SIZE
    d = np.zeros((H, W), dtype=np.float32)
    if points_xy.shape[0] == 0:
        return d
    pts = points_xy / DENSITY_STRIDE
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    np.add.at(d, (ys, xs), 1.0)
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(d, sigma=DENSITY_SIGMA)
    return d


class WheatCountDataset(Dataset):
    """新版叶尖计数数据集 (VOC XML, train/val 两个 split)。"""

    def __init__(self, root: str, split: str, train: bool):
        # 仅 train/val; 若调用方传 'test' 自动回退到 'val' (该集合无 test)
        if split == 'test':
            split = 'val'
        self.img_dir = os.path.join(root, 'images', split)
        self.ann_dir = os.path.join(root, 'annotations', split)
        self.files = sorted([f for f in os.listdir(self.img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        stem = os.path.splitext(f)[0]
        img = Image.open(os.path.join(self.img_dir, f))
        xml_path = os.path.join(self.ann_dir, stem + '.xml')
        points = _parse_voc_xml(xml_path) if os.path.isfile(xml_path) else \
                 np.zeros((0, 2), dtype=np.float32)
        x, pts_resized = adapt_cnt(img, points, self.train)
        density = _gaussian_density(pts_resized)
        return {
            'image':   x,
            'density': torch.from_numpy(density),
            'count':   torch.tensor(float(points.shape[0])),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'targets': {
                'density': torch.stack([b['density'] for b in batch]),
                'count':   torch.stack([b['count']   for b in batch]),
            },
        }
