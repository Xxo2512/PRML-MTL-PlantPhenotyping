from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class InputAdapter:
    """Normalize heterogeneous task samples into the frozen model contract."""

    def __init__(self, task: str, train: bool, input_size: int = 384):
        if task not in {"seg", "det", "cnt", "cls"}:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.train = train
        self.input_size = input_size

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image_pil"].convert("RGB")
        orig_w, orig_h = image.size

        if self.task == "det":
            image_tensor, scale, pad_left, pad_top = self._resize_keep_ratio_pad(image)
            targets = self._adapt_det_targets(sample, scale, pad_left, pad_top)
        elif self.task == "seg":
            image_tensor = self._resize_stretch(image)
            targets = self._adapt_seg_targets(sample)
        elif self.task == "cnt":
            image_tensor = self._resize_stretch(image)
            targets = self._adapt_cnt_targets(sample, orig_w, orig_h)
        else:
            image_tensor = self._resize_stretch(image)
            targets = self._adapt_cls_targets(sample)

        return {
            "image": self._normalize(image_tensor),
            "targets": targets,
            "meta": {
                "image_path": sample.get("image_path", ""),
                "orig_size": (orig_h, orig_w),
                "task": self.task,
            },
        }

    def _resize_stretch(self, image: Image.Image) -> torch.Tensor:
        tensor = self._pil_to_float_tensor(image).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0)

    def _resize_keep_ratio_pad(self, image: Image.Image) -> Tuple[torch.Tensor, float, int, int]:
        width, height = image.size
        scale = min(self.input_size / width, self.input_size / height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))

        tensor = self._pil_to_float_tensor(image).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)

        pad_left = (self.input_size - new_w) // 2
        pad_top = (self.input_size - new_h) // 2
        pad_right = self.input_size - new_w - pad_left
        pad_bottom = self.input_size - new_h - pad_top
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return tensor, scale, pad_left, pad_top

    def _adapt_det_targets(self, sample: Dict, scale: float, pad_left: int, pad_top: int) -> Dict:
        boxes = torch.as_tensor(sample.get("boxes", np.zeros((0, 4))), dtype=torch.float32)
        labels = torch.as_tensor(sample.get("labels", np.zeros((0,))), dtype=torch.long)
        if boxes.numel() > 0:
            boxes = boxes * scale
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            boxes = boxes.clamp(min=0, max=self.input_size)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
        return {"boxes": boxes, "labels": labels}

    def _adapt_seg_targets(self, sample: Dict) -> Dict:
        mask = torch.as_tensor(sample["mask"], dtype=torch.long).unsqueeze(0).unsqueeze(0).float()
        mask = F.interpolate(mask, size=(self.input_size, self.input_size), mode="nearest")
        return {"mask": mask.squeeze(0).squeeze(0).long()}

    def _adapt_cnt_targets(self, sample: Dict, orig_w: int, orig_h: int) -> Dict:
        points = torch.as_tensor(sample.get("points", np.zeros((0, 2))), dtype=torch.float32)
        if points.numel() > 0:
            points[:, 0] *= self.input_size / orig_w
            points[:, 1] *= self.input_size / orig_h
        count = int(sample.get("count", len(points)))
        return {"points": points, "count": torch.tensor(count, dtype=torch.float32)}

    def _adapt_cls_targets(self, sample: Dict) -> Dict:
        return {"label": torch.tensor(int(sample["label"]), dtype=torch.long)}

    @staticmethod
    def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @staticmethod
    def _normalize(image: torch.Tensor) -> torch.Tensor:
        return (image - IMAGENET_MEAN.to(image)) / IMAGENET_STD.to(image)
