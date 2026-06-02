"""统一的图像归一化 / 增广 / resize。

每个 Dataset 的 __getitem__ 末尾调用本模块对应函数得到 384x384 张量。
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE = 384


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


def adapt_cls(pil: Image.Image, train: bool) -> torch.Tensor:
    """分类: resize 到 INPUT_SIZE，可选水平翻转。"""
    pil = pil.convert('RGB').resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    if train and np.random.rand() < 0.5:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    return _normalize(TF.to_tensor(pil))


def adapt_seg(pil_img: Image.Image, pil_mask: Image.Image, train: bool):
    """分割: 同步缩放 image + mask 到 INPUT_SIZE。"""
    pil_img  = pil_img.convert('RGB').resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    pil_mask = pil_mask.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
    if train and np.random.rand() < 0.5:
        pil_img  = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    img = _normalize(TF.to_tensor(pil_img))
    mask = torch.from_numpy(np.array(pil_mask, dtype=np.int64))
    return img, mask


def adapt_det(pil: Image.Image, boxes_xyxy_norm: np.ndarray, train: bool):
    """检测: keep-ratio resize 到 INPUT_SIZE 的最长边并 pad，box 同步变换。

    boxes_xyxy_norm: (N, 4) in [0,1] 相对于原图。
    返回 boxes_xyxy_norm 仍然归一化到 INPUT_SIZE。
    """
    pil = pil.convert('RGB')
    W, H = pil.size
    s = INPUT_SIZE / max(W, H)
    new_w, new_h = int(round(W * s)), int(round(H * s))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(pil, (0, 0))
    if boxes_xyxy_norm.size > 0:
        b = boxes_xyxy_norm.copy()
        b[:, [0, 2]] *= W * s / INPUT_SIZE
        b[:, [1, 3]] *= H * s / INPUT_SIZE
        boxes_xyxy_norm = b
    if train and np.random.rand() < 0.5:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes_xyxy_norm.size > 0:
            x1 = 1 - boxes_xyxy_norm[:, 2]
            x2 = 1 - boxes_xyxy_norm[:, 0]
            boxes_xyxy_norm[:, 0] = x1
            boxes_xyxy_norm[:, 2] = x2
    return _normalize(TF.to_tensor(canvas)), boxes_xyxy_norm


def adapt_cnt(pil: Image.Image, points_xy: np.ndarray, train: bool):
    """计数: 同 det 的 keep-ratio 思路, 点坐标缩放到 [0, INPUT_SIZE)。"""
    pil = pil.convert('RGB')
    W, H = pil.size
    s = INPUT_SIZE / max(W, H)
    new_w, new_h = int(round(W * s)), int(round(H * s))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    canvas.paste(pil, (0, 0))
    pts = points_xy.astype(np.float32) * s if points_xy.size > 0 else points_xy
    if train and np.random.rand() < 0.5:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        if pts.size > 0:
            pts = pts.copy()
            pts[:, 0] = INPUT_SIZE - 1 - pts[:, 0]
    return _normalize(TF.to_tensor(canvas)), pts
