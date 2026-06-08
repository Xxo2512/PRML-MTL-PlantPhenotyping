from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

from datasets.input_adapter import InputAdapter
from datasets.wheat_det import WheatDetectionDataset
from models.heads.det_fcos import FCOSDetectionHead
from models.mtl.tadformer import TADFormerModel
from utils.collate import task_collate


class TinyBackbone(nn.Module):
    out_channels = [16, 32, 64, 128]
    out_strides = [4, 8, 16, 32]

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, 16, 3, stride=4, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
            ]
        )

    def forward(self, x, task=None):
        feats = {}
        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            feats[f"s{idx}"] = x
        return feats


def synthetic_adapter_check() -> None:
    image = Image.new("RGB", (1024, 512), color=(120, 160, 90))
    sample = {
        "image_path": "synthetic.png",
        "image_pil": image,
        "boxes": np.asarray([[100, 50, 300, 200], [700, 100, 900, 450]], dtype=np.float32),
        "labels": np.asarray([1, 1], dtype=np.int64),
    }
    adapted = InputAdapter("det", train=True)(sample)
    assert tuple(adapted["image"].shape) == (3, 384, 384)
    assert adapted["targets"]["boxes"].shape == (2, 4)
    assert adapted["targets"]["boxes"].min() >= 0
    assert adapted["targets"]["boxes"].max() <= 384
    print("adapter: ok", adapted["image"].shape, adapted["targets"]["boxes"].shape)


def model_forward_backward_check() -> None:
    backbone = TinyBackbone()
    head = FCOSDetectionHead(in_channels=backbone.out_channels, num_classes=1)
    cfg = {
        "model": {"freeze_backbone": False, "tadformer": {"task_dim": 16, "per_stage": True}},
        "tasks": {"det": {"enabled": True}},
    }
    model = TADFormerModel(cfg=cfg, backbone=backbone, heads={"det": head})
    model.train()

    batch = {
        "det": {
            "image": torch.randn(2, 3, 384, 384),
            "targets": {
                "boxes": [
                    torch.tensor([[20.0, 30.0, 120.0, 160.0]], dtype=torch.float32),
                    torch.tensor([[200.0, 100.0, 260.0, 180.0]], dtype=torch.float32),
                ],
                "labels": [
                    torch.tensor([1], dtype=torch.long),
                    torch.tensor([1], dtype=torch.long),
                ],
            },
        }
    }
    out = model(batch)
    loss = out["det"]["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    print("model: ok", float(loss.detach().cpu()), out["det"]["loss_items"])


def real_dataset_check(data_root: Path) -> None:
    dataset = WheatDetectionDataset(
        root=data_root,
        split="train",
        transform=InputAdapter("det", train=True),
    )
    subset = Subset(dataset, range(min(2, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        collate_fn=lambda batch: task_collate(batch, "det"),
    )
    batch = next(iter(loader))
    assert tuple(batch["image"].shape[1:]) == (3, 384, 384)
    assert len(batch["targets"]["boxes"]) == len(subset)
    print("dataset: ok", len(dataset), batch["image"].shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    synthetic_adapter_check()
    model_forward_backward_check()
    if args.data_root is not None:
        real_dataset_check(args.data_root)


if __name__ == "__main__":
    main()
