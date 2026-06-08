"""统一评测指标模块 — 覆盖 4 个任务的全部指标.

Per api_contract.md §7, 提供:
  - compute_cls_metric: 分类 — Acc / mAP / BA
  - compute_seg_metric: 分割 — mIoU / mAcc   (C 模块占位)
  - compute_det_metric: 检测 — AP / AP50      (B 模块占位)
  - compute_cnt_metric: 计数 — MAE / RMSE / R² (D 模块占位)

指标说明:
  - Acc  (Accuracy):         正确预测样本数 / 总样本数
  - mAP  (mean Average Precision): 各类别 AP 均值, 宏平均
  - BA   (Balanced Accuracy): 各类别召回率均值, 对不平衡数据更鲁棒
  - mIoU (mean Intersection over Union): 各类别 IoU 均值
  - mAcc (mean Accuracy):     各类别像素准确率均值
  - AP / AP50:                COCO 风格平均精度 (IoU 0.5:0.95 / IoU 0.5)
  - MAE (Mean Absolute Error):  |pred - gt| 均值
  - RMSE (Root Mean Square Error): sqrt(mean((pred - gt)²))
  - R²  (Coefficient of Determination): 1 - SS_res / SS_tot
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 分类指标 (E 模块)
# ══════════════════════════════════════════════════════════════════════


def compute_cls_metric(
    logits: torch.Tensor,
    labels: torch.Tensor,
    preds: Optional[torch.Tensor] = None,
    num_classes: int = 6,
) -> Dict[str, float]:
    """计算分类任务的 Acc / mAP / BA.

    Args:
        logits: [B, C] 或 [B, K-1] (ordinal) — 模型原始输出.
        labels: [B] — 真实类别标签 (0 .. C-1).
        preds:  [B] — 预测类别标签 (可选; 为 None 时用 argmax).
        num_classes: 类别总数.

    Returns:
        {'cls/acc': float, 'cls/mAP': float, 'cls/BA': float}
    """
    if preds is None:
        if logits.shape[-1] == num_classes:
            # CE 模式
            preds = logits.argmax(dim=-1)
        else:
            # Ordinal 模式: [B, K-1]
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=-1)

    labels = labels.long()
    preds = preds.long()

    # ---- Accuracy ----
    correct = (preds == labels).sum().item()
    total = labels.numel()
    acc = correct / max(total, 1)

    # ---- Balanced Accuracy (BA) ----
    # 各类别召回率的宏平均
    recalls = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            r = (preds[mask] == c).sum().item() / mask.sum().item()
            recalls.append(r)
    ba = sum(recalls) / max(len(recalls), 1)

    # ---- mean Average Precision (mAP) ----
    # 对每个类别计算 AP (PR 曲线下面积), 然后宏平均
    aps = []
    for c in range(num_classes):
        # 二值标签: 是否属于类别 c
        binary_labels = (labels == c).float()            # [B]
        # 置信度: softmax 后的第 c 维概率 (ordinal 下使用 predict_proba 类似逻辑)
        if logits.shape[-1] == num_classes:
            scores = F.softmax(logits, dim=-1)[:, c]      # [B]
        else:
            # Ordinal: 从 P(y > k) 推导 P(y = c)
            scores = _ordinal_class_probs(logits, num_classes)[:, c]
        ap = _compute_ap(scores, binary_labels)
        aps.append(ap)
    mAP = sum(aps) / max(len(aps), 1)

    return {
        "cls/acc": round(acc, 6),
        "cls/mAP": round(mAP, 6),
        "cls/BA": round(ba, 6),
    }


def _ordinal_class_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """从 ordinal logits [B, K-1] 推导类别概率 [B, K].

    P(y = 0)     = 1 - P(y > 0)
    P(y = k)     = P(y > k-1) - P(y > k)     (1 <= k <= K-2)
    P(y = K-1)   = P(y > K-2)
    """
    probs_greater = torch.sigmoid(logits)                   # [B, K-1]
    B = logits.shape[0]
    prob = torch.zeros(B, num_classes, device=logits.device)
    prob[:, 0] = 1.0 - probs_greater[:, 0]
    for k in range(1, num_classes - 1):
        prob[:, k] = probs_greater[:, k - 1] - probs_greater[:, k]
    prob[:, num_classes - 1] = probs_greater[:, num_classes - 2]
    prob = prob.clamp(min=1e-7)
    prob = prob / prob.sum(dim=1, keepdim=True)
    return prob


def _compute_ap(scores: torch.Tensor, binary_labels: torch.Tensor) -> float:
    """计算单类别 Average Precision.

    按置信度降序排列样本, 计算逐阈值 precision/recall,
    用 all-point interpolation (Pascal VOC 风格) 求 PR 曲线下面积.

    Args:
        scores: [B] — 属于该类别的置信度.
        binary_labels: [B] — 0/1 二值标签.

    Returns:
        AP 值, 范围 [0, 1].
    """
    if binary_labels.sum() == 0:
        return 0.0

    # 按分数降序排列
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = binary_labels[sorted_indices]

    # 累积 true positives 和 false positives
    tp = sorted_labels.cumsum(dim=0)
    fp = (1 - sorted_labels).cumsum(dim=0)

    # Precision 和 Recall
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / tp[-1].clamp(min=1)

    # All-point interpolation: 对每个 recall 点, 取右侧最大 precision
    # 两端填充避免边界效应
    precision_padded = torch.cat([
        torch.zeros(1, device=precision.device),
        precision,
        torch.zeros(1, device=precision.device),
    ])
    recall_padded = torch.cat([
        torch.zeros(1, device=recall.device),
        recall,
        torch.ones(1, device=recall.device),
    ])

    # 从后向前取 max
    for i in range(len(precision_padded) - 2, -1, -1):
        precision_padded[i] = torch.max(precision_padded[i], precision_padded[i + 1])

    # 计算面积: Σ Δr * p(r)
    recall_diff = recall_padded[1:] - recall_padded[:-1]
    ap = (precision_padded[:-1] * recall_diff).sum()
    return ap.item()


# ══════════════════════════════════════════════════════════════════════
# 分割指标 (C 模块占位 — 当前提供简化实现)
# ══════════════════════════════════════════════════════════════════════


def compute_seg_metric(
    pred_logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """计算分割 mIoU / mAcc.

    Args:
        pred_logits: [B, C, H, W] — 原始 logits.
        mask:        [B, H, W] — 真实标签 (long).
        num_classes: 类别数 (含背景).

    Returns:
        {'seg/mIoU': float, 'seg/mAcc': float}
    """
    pred = pred_logits.argmax(dim=1)                      # [B, H, W]
    mask = mask.long()

    ious = []
    accs = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (mask == c)
        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        iou = (intersection / union.clamp(min=1)).item()
        ious.append(iou)

        gt_c_count = gt_c.sum().float()
        acc = (intersection / gt_c_count.clamp(min=1)).item() if gt_c_count > 0 else 0.0
        accs.append(acc)

    return {
        "seg/mIoU": round(sum(ious) / max(len(ious), 1), 6),
        "seg/mAcc": round(sum(accs) / max(len(accs), 1), 6),
    }


# ══════════════════════════════════════════════════════════════════════
# 检测指标 (B 模块占位 — 当前提供简化 IoU 精度)
# ══════════════════════════════════════════════════════════════════════


def compute_det_metric(
    preds: Dict[str, torch.Tensor],
    boxes_gt: List[torch.Tensor],
    labels_gt: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """计算检测 AP / AP50 (简化版, 匹配用贪心 IoU).

    完整版应由 B 模块用 pycocotools 替换。

    Args:
        preds:    {'boxes': [B, N, 4], 'scores': [B, N], 'labels': [B, N]}.
        boxes_gt: List[[M_i, 4]] — 每张图的真实框.
        labels_gt: List[[M_i]] — 每张图的真实标签.
        iou_threshold: IoU 匹配阈值.

    Returns:
        {'det/AP50': float}
    """
    boxes_pred = preds.get("boxes")
    scores_pred = preds.get("scores")
    labels_pred = preds.get("labels")

    if boxes_pred is None:
        return {"det/AP50": 0.0, "det/AP": 0.0}

    # 简化: 按 score 排序, 贪心匹配, 计算 precision/recall
    all_scores = []
    all_matches = []
    n_gt_total = sum(len(b) for b in boxes_gt)

    for i in range(len(boxes_gt)):
        if i >= len(boxes_pred):
            break
        pb = boxes_pred[i]
        ps = scores_pred[i] if scores_pred is not None else torch.ones(len(pb))
        gb = boxes_gt[i]

        if len(gb) == 0:
            all_scores.extend(ps.tolist())
            all_matches.extend([False] * len(ps))
            continue

        # 贪心匹配 (按 score 降序)
        matched_gt = set()
        sorted_idx = torch.argsort(ps, descending=True)
        for idx in sorted_idx:
            best_iou = 0.0
            best_j = -1
            for j in range(len(gb)):
                if j in matched_gt:
                    continue
                iou = _compute_iou(pb[idx], gb[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j not in matched_gt:
                matched_gt.add(best_j)
                all_matches.append(True)
            else:
                all_matches.append(False)
            all_scores.append(ps[idx].item())

    if not all_scores:
        return {"det/AP50": 0.0, "det/AP": 0.0}

    # 按 score 降序计算 AP
    sorted_idx = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for idx in sorted_idx:
        if all_matches[idx]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / max(n_gt_total, 1))

    # All-point interpolation
    ap50 = _ap_from_pr(torch.tensor(precisions), torch.tensor(recalls))

    return {
        "det/AP50": round(ap50, 6),
        "det/AP": round(ap50, 6),  # 简化版 AP == AP50
    }


def _compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """计算两个框的 IoU (x1, y1, x2, y2 格式)."""
    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]).item() * (box_a[3] - box_a[1]).item()
    area_b = (box_b[2] - box_b[0]).item() * (box_b[3] - box_b[1]).item()
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ap_from_pr(precisions: torch.Tensor, recalls: torch.Tensor) -> float:
    """从 PR 曲线计算 AP (all-point interpolation)."""
    if len(precisions) == 0:
        return 0.0
    # 两端填充
    p = torch.cat([torch.zeros(1), precisions, torch.zeros(1)])
    r = torch.cat([torch.zeros(1), recalls, torch.ones(1)])
    for i in range(len(p) - 2, -1, -1):
        p[i] = torch.max(p[i], p[i + 1])
    ap = (p[:-1] * (r[1:] - r[:-1])).sum()
    return ap.item()


# ══════════════════════════════════════════════════════════════════════
# 计数指标 (D 模块占位)
# ══════════════════════════════════════════════════════════════════════


def compute_cnt_metric(
    pred_count: torch.Tensor,
    gt_count: torch.Tensor,
) -> Dict[str, float]:
    """计算计数 MAE / RMSE / R².

    Args:
        pred_count: [B] — 预测数量.
        gt_count:   [B] — 真实数量.

    Returns:
        {'cnt/MAE': float, 'cnt/RMSE': float, 'cnt/R2': float}
    """
    pred = pred_count.float()
    gt = gt_count.float()

    diff = pred - gt
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()

    # R² = 1 - SS_res / SS_tot
    ss_res = (diff ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()

    return {
        "cnt/MAE": round(mae, 6),
        "cnt/RMSE": round(rmse, 6),
        "cnt/R2": round(r2, 6),
    }


# ══════════════════════════════════════════════════════════════════════
# 聚合评测入口
# ══════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """全模型评测: 在所有 dataloader 上汇总 4 个任务的指标.

    Args:
        model:   MTLModel 子类.
        loaders: {'seg': DataLoader, 'det': ..., 'cnt': ..., 'cls': ...}.
        device:  计算设备.

    Returns:
        Flat dict 汇总所有指标, 如:
        {'cls/acc': 0.85, 'cls/mAP': 0.82, 'cls/BA': 0.80,
         'seg/mIoU': 0.62, 'det/AP50': 0.71, 'cnt/MAE': 2.3, ...,
         'aggregate': 0.xxx}
    """
    model.eval()
    model.to(device)

    all_metrics: Dict[str, List[float]] = {}

    with torch.no_grad():
        for task, loader in loaders.items():
            for batch in loader:
                # 移入设备
                batch = _to_device(batch, device)

                # Forward
                out = model({task: batch})
                task_out = out.get(task, {})

                if "pred" not in task_out:
                    continue

                # 提取指标 (仅处理当前 task)
                targets = batch.get("targets", {})
                metrics = _compute_single_task_metrics(task, task_out, targets)

                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)

    # 平均 (跨 batch)
    averaged = {k: sum(v) / len(v) for k, v in all_metrics.items()}

    # 计算 total aggregate (seg/mIoU + det/AP50 + -cnt/MAE + cls/mAP) / 4
    aggregate_parts = []
    if "seg/mIoU" in averaged:
        aggregate_parts.append(averaged["seg/mIoU"])
    if "det/AP50" in averaged:
        aggregate_parts.append(averaged["det/AP50"])
    if "cnt/MAE" in averaged:
        aggregate_parts.append(-averaged["cnt/MAE"] / 10.0)  # 归一化
    if "cls/mAP" in averaged:
        aggregate_parts.append(averaged["cls/mAP"])
    if aggregate_parts:
        averaged["aggregate"] = round(sum(aggregate_parts) / len(aggregate_parts), 6)

    model.train()
    return averaged


def _compute_single_task_metrics(
    task: str, task_out: Dict, targets: Dict
) -> Dict[str, float]:
    """单任务单 batch 指标计算."""
    if task == "cls":
        logits = task_out["pred"]
        labels = targets.get("label")
        if labels is not None:
            if hasattr(task_out.get("head"), "predict_class"):
                preds = task_out["head"].predict_class(logits)
            else:
                preds = logits.argmax(dim=-1) if logits.shape[-1] > 1 else (torch.sigmoid(logits) > 0.5).sum(dim=-1)
            return compute_cls_metric(logits, labels, preds)

    elif task == "seg":
        logits = task_out["pred"]
        mask = targets.get("mask")
        if mask is not None:
            return compute_seg_metric(logits, mask)

    elif task == "det":
        preds = task_out["pred"]
        boxes_gt = targets.get("boxes", [])
        labels_gt = targets.get("labels", [])
        if boxes_gt is not None:
            return compute_det_metric(preds, boxes_gt, labels_gt)

    elif task == "cnt":
        pred = task_out["pred"]
        if isinstance(pred, dict):
            count_pred = pred.get("count")
        else:
            count_pred = pred
        count_gt = targets.get("count")
        if count_pred is not None and count_gt is not None:
            return compute_cnt_metric(count_pred, count_gt)

    return {}


def _to_device(obj, device: torch.device):
    """递归将 batch 数据移入设备."""
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    return obj
