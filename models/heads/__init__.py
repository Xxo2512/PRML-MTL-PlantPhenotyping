from .base import BaseTaskHead
from .cls_mlp import ClsHead
from .cls_pgt import ClsPGTHead
from .cnt_pet import CntHead
from .det_fcos import DetHead, FCOSDetectionHead
from .seg_dpt import SegHead


def build_head(task: str, cfg):
    """Build task head.

    When the method is 'pgt', cls task uses ClsPGTHead (supports CE + ordinal).
    Otherwise falls back to ClsHead (CE only).
    """
    if task == "cls":
        method = getattr(cfg, "method", "vanilla")
        num_classes = cfg.tasks.cls.num_classes
        if method == "pgt":
            loss_type = getattr(cfg.tasks.cls, "loss", "ce")
            return ClsPGTHead(in_dim=768, num_classes=num_classes, loss_type=loss_type)
        return ClsHead(in_dim=768, num_classes=num_classes)
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
    "ClsPGTHead",
    "CntHead",
    "DetHead",
    "FCOSDetectionHead",
    "SegHead",
    "build_head",
]
