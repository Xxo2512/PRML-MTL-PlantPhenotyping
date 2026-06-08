"""4 个任务的评测 metric + 统一 evaluate 入口。

数学定义:
  seg: mIoU = mean_c IoU_c,  IoU_c = TP_c / (TP_c + FP_c + FN_c)
       mAcc = mean_c TP_c / (TP_c + FN_c)
  det (简化版, W14 升真 AP/AP50):
       pos_recall = #正确预测中心 / #GT box  (在 stride-8 上, sigmoid(obj)>0.5 即正样本)
       pos_precision = #正确预测中心 / #预测正样本
       (用同一组阈值近似 PR; W14 由 B 替换为 pycocotools AP)
  cnt: 把预测密度图求和得 count_pred, 比 GT count
       MAE  = mean |c_pred - c_gt|
       RMSE = sqrt(mean (c_pred - c_gt)^2)
       R²   = 1 - SS_res / SS_tot
  cls: top-1 acc, macro F1; 生育期是有序 (6 类) 故额外报 acc@±1
       acc@±1 = mean( |y_pred - y_gt| <= 1 )
"""
from __future__ import annotations
from typing import Dict, List
import math
import numpy as np
import torch
import torch.nn.functional as F


# =========================================================================
# Seg
# =========================================================================
class SegMetric:
    def __init__(self, num_classes: int):
        self.K = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred_logits: torch.Tensor, mask: torch.Tensor):
        # pred_logits: [B,C,H,W];  mask: [B,H,W]
        pred = pred_logits.argmax(1)
        p = pred.flatten().cpu().numpy()
        g = mask.flatten().cpu().numpy()
        valid = (g >= 0) & (g < self.K)
        idx = self.K * g[valid] + p[valid]
        binc = np.bincount(idx, minlength=self.K * self.K)
        self.cm += binc.reshape(self.K, self.K)

    def compute(self) -> Dict[str, float]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fn = cm.sum(1) - tp
        fp = cm.sum(0) - tp
        iou = tp / (tp + fp + fn + 1e-9)
        acc = tp / (cm.sum(1) + 1e-9)
        return {'seg/mIoU': float(iou.mean()), 'seg/mAcc': float(acc.mean())}


# =========================================================================
# Det (W13 简化版, W14 升真 AP)
# =========================================================================
class DetMetric:
    def __init__(self, score_thr: float = 0.5, input_size: int = 384, stride: int = 8):
        self.t = score_thr
        self.S = input_size
        self.r = stride
        self.tp = 0; self.fp = 0; self.fn = 0

    def update(self, pred: dict, boxes_list: List[torch.Tensor]):
        # pred['obj_logit']: [B,1,H,W];   boxes_list: B 个 [N_i,4] in [0,1] xyxy
        obj = torch.sigmoid(pred['obj_logit']) > self.t
        for b in range(obj.size(0)):
            n_pred = int(obj[b].sum().item())
            gt = boxes_list[b]
            n_gt = gt.shape[0]
            # 用"GT 中心落在某个被预测为正的 cell"作为 hit
            if n_gt > 0:
                cx = ((gt[:, 0] + gt[:, 2]) / 2 * obj.shape[-1]).clamp(0, obj.shape[-1] - 1).long()
                cy = ((gt[:, 1] + gt[:, 3]) / 2 * obj.shape[-2]).clamp(0, obj.shape[-2] - 1).long()
                hit = obj[b, 0, cy, cx].sum().item()
            else:
                hit = 0
            self.tp += int(hit)
            self.fn += int(n_gt - hit)
            self.fp += int(max(0, n_pred - hit))

    def compute(self) -> Dict[str, float]:
        prec = self.tp / (self.tp + self.fp + 1e-9)
        rec  = self.tp / (self.tp + self.fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        return {'det/precision': float(prec), 'det/recall': float(rec), 'det/f1': float(f1)}


# =========================================================================
# Cnt
# =========================================================================
class CntMetric:
    def __init__(self):
        self.preds: List[float] = []
        self.gts: List[float] = []

    def update(self, pred: dict, gt_count: torch.Tensor):
        p = pred['count'].detach().float().cpu().tolist()
        g = gt_count.float().cpu().tolist()
        self.preds.extend(p); self.gts.extend(g)

    def compute(self) -> Dict[str, float]:
        p = np.array(self.preds); g = np.array(self.gts)
        if p.size == 0:
            return {'cnt/MAE': 0.0, 'cnt/RMSE': 0.0, 'cnt/R2': 0.0}
        err = p - g
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        ss_res = float(((g - p) ** 2).sum())
        ss_tot = float(((g - g.mean()) ** 2).sum()) + 1e-9
        r2 = 1.0 - ss_res / ss_tot
        return {'cnt/MAE': mae, 'cnt/RMSE': rmse, 'cnt/R2': float(r2)}


# =========================================================================
# Cls
# =========================================================================
class ClsMetric:
    def __init__(self, num_classes: int):
        self.K = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred_logits: torch.Tensor, label: torch.Tensor):
        p = pred_logits.argmax(1).cpu().numpy()
        g = label.cpu().numpy()
        for gt, pr in zip(g, p):
            self.cm[gt, pr] += 1

    def compute(self) -> Dict[str, float]:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp
        acc = tp.sum() / (cm.sum() + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        # 有序 ±1 容忍准确率: |y_pred - y_gt| <= 1
        n_total = cm.sum()
        adj = 0.0
        for gt in range(self.K):
            for pr in range(self.K):
                if abs(gt - pr) <= 1:
                    adj += cm[gt, pr]
        return {
            'cls/top1':      float(acc),
            'cls/macro_f1':  float(np.nan_to_num(f1).mean()),
            'cls/acc_adj1':  float(adj / (n_total + 1e-9)),
        }


# =========================================================================
# 统一入口
# =========================================================================
@torch.no_grad()
def evaluate_model(model, val_loaders: Dict[str, 'DataLoader'], cfg, device) -> Dict[str, float]:
    model.eval()
    metrics = {}
    if 'seg' in val_loaders: metrics['seg'] = SegMetric(cfg.tasks.seg.num_classes)
    if 'det' in val_loaders: metrics['det'] = DetMetric()
    if 'cnt' in val_loaders: metrics['cnt'] = CntMetric()
    if 'cls' in val_loaders: metrics['cls'] = ClsMetric(cfg.tasks.cls.num_classes)

    def mv(x):
        if isinstance(x, dict): return {k: mv(v) for k, v in x.items()}
        if isinstance(x, list): return [mv(y) for y in x]
        if torch.is_tensor(x): return x.to(device, non_blocking=True)
        return x

    for task, loader in val_loaders.items():
        for batch in loader:
            batch = mv(batch)
            out = model({task: batch})[task]
            pred = out['pred']
            tgt = batch['targets']
            if task == 'seg':
                metrics['seg'].update(pred, tgt['mask'])
            elif task == 'det':
                metrics['det'].update(pred, tgt['boxes'])
            elif task == 'cnt':
                metrics['cnt'].update(pred, tgt['count'])
            elif task == 'cls':
                metrics['cls'].update(pred, tgt['label'])

    flat = {}
    for m in metrics.values():
        flat.update(m.compute())
    # 主指标的"聚合"分: 取每任务的"越大越好"指标平均 (cnt 用 1/(1+MAE) 转正)
    agg_parts = []
    if 'seg/mIoU' in flat:  agg_parts.append(flat['seg/mIoU'])
    if 'det/f1'   in flat:  agg_parts.append(flat['det/f1'])
    if 'cnt/MAE'  in flat:  agg_parts.append(1.0 / (1.0 + flat['cnt/MAE']))
    if 'cls/top1' in flat:  agg_parts.append(flat['cls/top1'])
    if agg_parts:
        flat['aggregate'] = float(sum(agg_parts) / len(agg_parts))
    return flat
