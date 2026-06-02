"""分类: 6 类小麦生育期。ImageFolder 风格。"""
from __future__ import annotations
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from .input_adapter import adapt_cls

CLASSES = ['1_Tillering', '2_Jointing', '3_BH', '4_Flowering', '5_Filling', '6_Ripening']
NAME2IDX = {c: i for i, c in enumerate(CLASSES)}


class WheatGrowthStageDataset(Dataset):
    def __init__(self, root: str, split: str, train: bool):
        self.split_dir = os.path.join(root, split)
        self.samples: list[tuple[str, int]] = []
        for cname in CLASSES:
            cdir = os.path.join(self.split_dir, cname)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cdir, f), NAME2IDX[cname]))
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        x = adapt_cls(img, self.train)
        return {'image': x, 'label': torch.tensor(label, dtype=torch.long)}

    @staticmethod
    def collate_fn(batch):
        return {
            'image': torch.stack([b['image'] for b in batch]),
            'targets': {'label': torch.stack([b['label'] for b in batch])},
        }
