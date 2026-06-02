"""分割: 4 类（含背景）小麦器官分割。"""
from __future__ import annotations
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from .input_adapter import adapt_seg


class WheatOrganSegDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool):
        self.img_dir  = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'class_id')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith('.png')])
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, f))
        mask = Image.open(os.path.join(self.mask_dir, f))
        x, m = adapt_seg(img, mask, self.train)
        return {'image': x, 'mask': m}

    @staticmethod
    def collate_fn(batch):
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'targets': {'mask': torch.stack([b['mask'] for b in batch])},
        }
