from .base import BaseTaskHead
from .cls_mlp import ClsHead
from .cnt_pet import CntHead
from .det_fcos import DetHead, FCOSDetectionHead
from .seg_dpt import SegHead


def build_head(task: str, cfg):
    if task == "cls":
        return ClsHead(in_dim=768, num_classes=cfg.tasks.cls.num_classes)
    if task == "seg":
        return SegHead(num_classes=cfg.tasks.seg.num_classes)
    if task == "det":
        return DetHead()
    if task == "cnt":
        return CntHead()
    raise ValueError(task)


__all__ = [
    "BaseTaskHead",
    "ClsHead",
    "CntHead",
    "DetHead",
    "FCOSDetectionHead",
    "SegHead",
    "build_head",
]
