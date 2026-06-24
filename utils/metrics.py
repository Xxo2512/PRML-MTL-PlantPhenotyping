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
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """计算检测 AP / AP50 (COCO 风格 10 阈值平均, 贪心匹配).

    向量化版: 每图 IoU 矩阵在 GPU/torch 上一次性算完, 再转 numpy 跑匹配 inner loop.
    比逐对 .item() 调 _compute_iou 快 10-30 倍 (避免大量 GPU→CPU sync).

    Args:
        preds:    {'boxes': [B, N, 4], 'scores': [B, N], 'labels': [B, N]}.
        boxes_gt: List[[M_i, 4]] — 每张图的真实框.
        labels_gt: List[[M_i]] — 每张图的真实标签.
        iou_thresholds: IoU 阈值序列, 默认 COCO 风格 [.5:.05:.95]; 第 0 项作为 AP50.

    Returns:
        {'det/AP50': float, 'det/AP': float}
    """
    import numpy as np

    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    boxes_pred = preds.get("boxes")
    scores_pred = preds.get("scores")
    if boxes_pred is None:
        return {"det/AP50": 0.0, "det/AP": 0.0}

    n_gt_total = sum(len(b) for b in boxes_gt)
    T = len(iou_thresholds)
    thr_arr = np.asarray(iou_thresholds, dtype=np.float32)

    # 每阈值累积 (scores, matches)
    matches_per_thr: List[List[bool]] = [[] for _ in range(T)]
    scores_per_thr: List[List[float]] = [[] for _ in range(T)]

    for i in range(len(boxes_gt)):
        if i >= len(boxes_pred):
            break
        pb = boxes_pred[i]
        ps = scores_pred[i] if scores_pred is not None else torch.ones(len(pb))
        gb = boxes_gt[i]

        if len(pb) == 0:
            continue
        ps_np = ps.detach().cpu().numpy() if torch.is_tensor(ps) else np.asarray(ps)
        sorted_idx = np.argsort(-ps_np)

        if len(gb) == 0:
            # 全 FP
            for t_idx in range(T):
                scores_per_thr[t_idx].extend(ps_np[sorted_idx].tolist())
                matches_per_thr[t_idx].extend([False] * len(sorted_idx))
            continue

        # ---- 一次性算 IoU 矩阵 [N, M] (numpy) ----
        iou_mat = _box_iou_matrix(pb, gb).detach().cpu().numpy()

        # ---- 每阈值: 贪心匹配 (numpy inner loop, 无 GPU sync) ----
        M = len(gb)
        for t_idx in range(T):
            thr = float(thr_arr[t_idx])
            matched_gt = np.zeros(M, dtype=bool)
            for idx in sorted_idx:
                # 在未匹配 GT 中找最大 IoU
                ious_row = iou_mat[idx]
                ious_masked = np.where(matched_gt, -1.0, ious_row)
                best_j = int(np.argmax(ious_masked))
                best_iou = float(ious_masked[best_j])
                if best_iou >= thr:
                    matched_gt[best_j] = True
                    matches_per_thr[t_idx].append(True)
                else:
                    matches_per_thr[t_idx].append(False)
                scores_per_thr[t_idx].append(float(ps_np[idx]))

    # ---- 每阈值算 AP ----
    per_thr_ap: List[float] = []
    for t_idx in range(T):
        scores = scores_per_thr[t_idx]
        matches = matches_per_thr[t_idx]
        if not scores:
            per_thr_ap.append(0.0)
            continue
        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        tp_cum = 0
        fp_cum = 0
        precisions: List[float] = []
        recalls: List[float] = []
        for idx in order:
            if matches[idx]:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / max(n_gt_total, 1))
        per_thr_ap.append(_ap_from_pr(torch.tensor(precisions), torch.tensor(recalls)))

    ap50 = per_thr_ap[0] if per_thr_ap else 0.0
    ap = sum(per_thr_ap) / max(len(per_thr_ap), 1)
    return {
        "det/AP50": round(ap50, 6),
        "det/AP": round(ap, 6),
    }


def _box_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """计算两组框的 IoU 矩阵 [N, M] (xyxy 格式, 任意 device).

    向量化, 无 .item() 调用.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return torch.zeros(len(boxes_a), len(boxes_b))
    x1 = torch.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    a_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    union = a_a[:, None] + a_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


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
    """全模型评测: 在 loaders 上跨 batch 累积 pred/gt, 再在全集上一次性算指标.

    旧版本是 "每 batch 算指标后跨 batch 平均", 这对 cls/mAP, cnt/R², det/AP
    都是错的 (per-class AP / 总平方和都要全集统计). 改成 accumulate-then-compute.

    Args:
        model:   MTLModel 子类.
        loaders: {'seg': DataLoader, 'det': ..., 'cnt': ..., 'cls': ...},
                 调用者可只传子集 (single-task 评测).
        device:  计算设备.

    Returns:
        Flat dict, 如 {'cls/acc': 0.85, 'seg/mIoU': 0.62, ..., 'aggregate': ...}.
        若某 task 不在 loaders 里, 对应键不会出现.
    """
    model.eval()
    model.to(device)

    cls_logits: List[torch.Tensor] = []
    cls_labels: List[torch.Tensor] = []
    seg_inter = None   # [C] tensor
    seg_union = None   # [C] tensor
    seg_gt_count = None  # [C] tensor (per-class pixel count for mAcc)
    seg_correct_count = None  # [C] tensor (per-class correct pixel count)
    seg_num_classes = None
    det_preds_acc: List[Dict[str, torch.Tensor]] = []
    det_boxes_gt_acc: List[torch.Tensor] = []
    det_labels_gt_acc: List[torch.Tensor] = []
    cnt_pred: List[torch.Tensor] = []
    cnt_gt: List[torch.Tensor] = []

    with torch.no_grad():
        for task, loader in loaders.items():
            for batch in loader:
                batch = _to_device(batch, device)
                out = model({task: batch})
                task_out = out.get(task, {})
                if "pred" not in task_out:
                    continue
                targets = batch.get("targets", {})

                if task == "cls":
                    cls_logits.append(task_out["pred"].detach().cpu())
                    cls_labels.append(targets["label"].detach().cpu())

                elif task == "seg":
                    pred = task_out["pred"].argmax(dim=1).cpu()      # [B,H,W]
                    mask = targets["mask"].long().cpu()
                    C = task_out["pred"].shape[1]
                    if seg_num_classes is None:
                        seg_num_classes = C
                        seg_inter = torch.zeros(C, dtype=torch.long)
                        seg_union = torch.zeros(C, dtype=torch.long)
                        seg_gt_count = torch.zeros(C, dtype=torch.long)
                        seg_correct_count = torch.zeros(C, dtype=torch.long)
                    for c in range(C):
                        pred_c = (pred == c)
                        gt_c = (mask == c)
                        seg_inter[c] += (pred_c & gt_c).sum()
                        seg_union[c] += (pred_c | gt_c).sum()
                        seg_gt_count[c] += gt_c.sum()
                        seg_correct_count[c] += (pred_c & gt_c).sum()

                elif task == "det":
                    pred_dict = task_out["pred"]
                    if isinstance(pred_dict, dict) and "boxes" in pred_dict:
                        B = len(pred_dict["boxes"])
                        for b in range(B):
                            det_preds_acc.append({
                                "boxes": pred_dict["boxes"][b].detach().cpu(),
                                "scores": pred_dict["scores"][b].detach().cpu()
                                          if "scores" in pred_dict else torch.ones(len(pred_dict["boxes"][b])),
                                "labels": pred_dict["labels"][b].detach().cpu()
                                          if "labels" in pred_dict else torch.zeros(len(pred_dict["boxes"][b]), dtype=torch.long),
                            })
                        bgt = targets.get("boxes", [])
                        lgt = targets.get("labels", [])
                        if isinstance(bgt, list):
                            for b in range(B):
                                det_boxes_gt_acc.append(bgt[b].detach().cpu() if torch.is_tensor(bgt[b]) else torch.as_tensor(bgt[b]))
                                det_labels_gt_acc.append(lgt[b].detach().cpu() if torch.is_tensor(lgt[b]) else torch.as_tensor(lgt[b]))
                        else:
                            for b in range(B):
                                det_boxes_gt_acc.append(bgt[b].detach().cpu())
                                det_labels_gt_acc.append(lgt[b].detach().cpu())

                elif task == "cnt":
                    pred = task_out["pred"]
                    count_pred = pred.get("count") if isinstance(pred, dict) else pred
                    count_gt = targets.get("count")
                    if count_pred is not None and count_gt is not None:
                        cnt_pred.append(count_pred.detach().cpu())
                        cnt_gt.append(count_gt.detach().cpu())

    metrics: Dict[str, float] = {}

    if cls_logits:
        logits = torch.cat(cls_logits, dim=0)
        labels = torch.cat(cls_labels, dim=0)
        num_classes = logits.shape[-1] if logits.shape[-1] > 1 else int(labels.max().item()) + 1
        metrics.update(compute_cls_metric(logits, labels, num_classes=num_classes))

    if seg_inter is not None:
        ious = (seg_inter.float() / seg_union.float().clamp(min=1)).tolist()
        accs = (seg_correct_count.float() / seg_gt_count.float().clamp(min=1)).tolist()
        metrics["seg/mIoU"] = round(sum(ious) / max(len(ious), 1), 6)
        metrics["seg/mAcc"] = round(sum(accs) / max(len(accs), 1), 6)

    if det_preds_acc:
        # Re-pack into batched dict structure expected by compute_det_metric
        det_pred_packed = {
            "boxes":  [p["boxes"] for p in det_preds_acc],
            "scores": [p["scores"] for p in det_preds_acc],
            "labels": [p["labels"] for p in det_preds_acc],
        }
        metrics.update(compute_det_metric(det_pred_packed, det_boxes_gt_acc, det_labels_gt_acc))

    if cnt_pred:
        pred = torch.cat(cnt_pred, dim=0)
        gt = torch.cat(cnt_gt, dim=0)
        metrics.update(compute_cnt_metric(pred, gt))

    # Aggregate (供主表排名用): seg/mIoU + det/AP50 + (-cnt/MAE/10) + cls/mAP, 均存在才参与
    parts: List[float] = []
    if "seg/mIoU" in metrics: parts.append(metrics["seg/mIoU"])
    if "det/AP50" in metrics: parts.append(metrics["det/AP50"])
    if "cnt/MAE" in metrics: parts.append(-metrics["cnt/MAE"] / 10.0)
    if "cls/mAP" in metrics: parts.append(metrics["cls/mAP"])
    if parts:
        metrics["aggregate"] = round(sum(parts) / len(parts), 6)

    model.train()
    return metrics


def _to_device(obj, device: torch.device):
    """递归将 batch 数据移入设备."""
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    return obj
