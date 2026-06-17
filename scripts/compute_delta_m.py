"""从 logs/results.csv 计算 Δm — MTL 综合性能指标 (Maninis et al. CVPR 2019).

公式:
  Δm = (1/T) Σ_t (-1)^l_t · (M_t^{MTL} - M_t^{STL}) / M_t^{STL}

  - T = 任务数 (本项目 = 4)
  - M_t = 任务 t 的指标值
  - l_t = 1 若该指标越低越好 (MAE/RMSE 等); 0 若越高越好 (mIoU/AP/Acc 等)
  - STL: single-task baseline (single_<task>_<ep>ep)
  - MTL: 当前评测方法

  Δm > 0  → MTL 综合优于单任务基线
  Δm < 0  → 综合落后

用法:
  python scripts/compute_delta_m.py                       # 默认 logs/results.csv, 默认 baseline 选 single_*
  python scripts/compute_delta_m.py --csv logs/v2.csv
  python scripts/compute_delta_m.py --baseline-prefix single_  --baseline-epoch 5ep
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, Optional


# 任务指标的方向: 'higher' 越高越好, 'lower' 越低越好
TASK_METRIC = [
    ("seg/mIoU", "higher"),
    ("det/AP50", "higher"),
    ("cnt/MAE",  "lower"),
    ("cls/acc",  "higher"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="logs/results.csv", help="results csv path")
    p.add_argument("--baseline-prefix", default="single_",
                   help="single-task baseline tag prefix (default: single_)")
    p.add_argument("--baseline-tags", default=None,
                   help="逗号分隔的 4 个 baseline tag, 用于显式指定. "
                        "未指定则按 prefix 自动找 single_<task>_*")
    return p.parse_args()


def read_results(csv_path: str) -> Dict[str, Dict[str, float]]:
    """返回 {tag: {metric_name: float, ...}}"""
    rows: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row["tag"]
            metrics = {}
            for k, v in row.items():
                if v in (None, "", "-"): continue
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
            rows[tag] = metrics
    return rows


def find_baseline_for_task(
    rows: Dict[str, Dict[str, float]], task_key: str, prefix: str
) -> Optional[str]:
    """对任务 task_key='seg/mIoU' 找一个 tag 以 prefix+<seg-style> 开头且含该指标的."""
    task_short = task_key.split("/")[0]  # 'seg'
    candidates = [
        tag for tag, m in rows.items()
        if tag.startswith(prefix + task_short + "_") and task_key in m
    ]
    if not candidates:
        return None
    # 取该 task 指标值"最高"的 (越好越好 → 拿到高的) 作为 baseline 上限
    # 若是 lower-is-better 指标, 此函数仍按 task_key 取数; 内部会用 metric value 判断
    # 但 STL 通常只跑一次, 直接选第一个即可
    return sorted(candidates)[0]


def compute_delta_m(
    method_metrics: Dict[str, float],
    baseline_per_task: Dict[str, float],
) -> Dict[str, float]:
    """给定 MTL 方法的指标 dict 和每任务的 STL 基线, 返回 per-task 相对变化 + Δm 平均."""
    per_task = {}
    for task_key, direction in TASK_METRIC:
        if task_key not in method_metrics or task_key not in baseline_per_task:
            per_task[task_key] = None
            continue
        mtl = method_metrics[task_key]
        stl = baseline_per_task[task_key]
        if stl == 0:
            per_task[task_key] = None
            continue
        if direction == "higher":
            delta = (mtl - stl) / abs(stl)
        else:
            delta = -(mtl - stl) / abs(stl)         # 等价 (stl - mtl) / stl
        per_task[task_key] = delta * 100.0          # %

    valid = [v for v in per_task.values() if v is not None]
    delta_m = sum(valid) / len(valid) if valid else None
    return {"per_task": per_task, "delta_m": delta_m}


def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        print(f"[err] csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = read_results(args.csv)
    print(f"[csv] loaded {len(rows)} rows from {args.csv}")

    # 找各任务的 STL baseline
    if args.baseline_tags:
        # 显式: --baseline-tags single_seg_5ep,single_det_5ep,single_cnt_5ep,single_cls_5ep
        baseline_tags = [t.strip() for t in args.baseline_tags.split(",")]
        if len(baseline_tags) != 4:
            print(f"[err] --baseline-tags 需要给 4 个 (seg/det/cnt/cls 顺序)", file=sys.stderr)
            sys.exit(1)
        baseline_per_task = {}
        for (task_key, _), tag in zip(TASK_METRIC, baseline_tags):
            if tag in rows and task_key in rows[tag]:
                baseline_per_task[task_key] = rows[tag][task_key]
            else:
                print(f"[warn] baseline tag={tag} 或指标 {task_key} 缺失", file=sys.stderr)
    else:
        baseline_per_task = {}
        for task_key, _ in TASK_METRIC:
            tag = find_baseline_for_task(rows, task_key, args.baseline_prefix)
            if tag and task_key in rows[tag]:
                baseline_per_task[task_key] = rows[tag][task_key]
                print(f"[baseline] {task_key:12s} ← {tag} = {rows[tag][task_key]:.4f}")
            else:
                print(f"[warn] 找不到 {task_key} 的 baseline (prefix={args.baseline_prefix})")

    if len(baseline_per_task) < 4:
        print("[err] 缺失 baseline, 无法算 Δm", file=sys.stderr)
        sys.exit(2)

    # 找所有 MTL 方法 (非 single_ 前缀的 row)
    print()
    print(f"{'method tag':<25s} | " + " | ".join(f"{k:12s}" for k, _ in TASK_METRIC) + " |   Δm")
    print("-" * 95)

    method_tags = [t for t in rows if not t.startswith(args.baseline_prefix)]
    for tag in sorted(method_tags):
        m = rows[tag]
        # 该方法必须 4 个 task 指标全有, 否则跳过
        if not all(k in m for k, _ in TASK_METRIC):
            continue
        result = compute_delta_m(m, baseline_per_task)
        cells = []
        for task_key, _ in TASK_METRIC:
            v = result["per_task"].get(task_key)
            cells.append(f"{v:+7.2f}%   " if v is not None else f"{'--':>12s}")
        dm = result["delta_m"]
        dm_str = f"{dm:+7.2f}%" if dm is not None else " --"
        print(f"{tag:<25s} | " + " | ".join(cells) + f"| {dm_str}")


if __name__ == "__main__":
    main()
