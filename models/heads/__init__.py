from .base import BaseTaskHead
from .cls_mlp import ClsHead
from .seg_dpt import SegHead
from .det_fcos import DetHead
from .cnt_pet import CntHead


def build_head(task: str, cfg):
    if task == 'cls':
        return ClsHead(in_dim=768, num_classes=cfg.tasks.cls.num_classes)
    if task == 'seg':
        return SegHead(num_classes=cfg.tasks.seg.num_classes)
    if task == 'det':
        return DetHead()
    if task == 'cnt':
        return CntHead()
    raise ValueError(task)


__all__ = ['BaseTaskHead', 'ClsHead', 'SegHead', 'DetHead', 'CntHead', 'build_head']
