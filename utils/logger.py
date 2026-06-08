"""统一日志与 checkpoint 管理 — TensorBoard + CSV + 模型保存.

Per api_contract.md §9 规范:
  Checkpoint 格式:
    {
        'step': int,
        'epoch': int,
        'model_state': dict,
        'optimizer_state': dict,
        'scheduler_state': dict,
        'loss_agg_state': dict,
        'best_metric': dict,
        'cfg': dict,
    }

  Checkpoint 命名:
    - 按 step 保存: <out_dir>/ckpt_step{step}.pt
    - 最优模型:     <out_dir>/best.pt
    - 最近模型:     <out_dir>/last.pt

  CSV 格式 (logs/results.csv):
    step, epoch, cls/loss, seg/loss, det/loss, cnt/loss, total_loss,
    cls/acc, cls/mAP, cls/BA, seg/mIoU, seg/mAcc, det/AP50, det/AP,
    cnt/MAE, cnt/RMSE, cnt/R2, aggregate, lr, timestamp

用法:
  from utils.logger import TBLogger, CheckpointManager, CSVLogger

  # TensorBoard
  tb = TBLogger(log_dir='logs/pgt_swint_384_rr')

  # CSV
  csv = CSVLogger(log_dir='logs/pgt_swint_384_rr')

  # Checkpoint
  ckpt = CheckpointManager(out_dir='checkpoints/pgt_swint_384_rr')
"""
from __future__ import annotations
import csv
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# TensorBoard Logger
# ══════════════════════════════════════════════════════════════════════


class TBLogger:
    """TensorBoard 日志记录器.

    记录内容:
      - 各任务 loss (scalar): cls/loss, seg/loss, det/loss, cnt/loss, total_loss
      - 各任务指标 (scalar): cls/acc, cls/mAP, cls/BA, seg/mIoU, ...
      - 学习率 (scalar): lr
      - 可选: 模型图, 梯度直方图

    Args:
        log_dir: TensorBoard 日志目录.
        enabled: 是否启用 (无 tensorboard 时自动降级).
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[TBLogger] tensorboard not available, logging disabled")
                self.enabled = False

    # ---- 标量记录 ----

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量指标."""
        if self.enabled and self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: Dict[str, float], step: int) -> None:
        """记录多个标量 (共享同一图表区域)."""
        if self.enabled and self._writer is not None:
            self._writer.add_scalars(main_tag, tag_value_dict, step)

    def log_losses(self, per_task_loss: Dict[str, float], total_loss: float, step: int) -> None:
        """记录 loss 面板.

        Args:
            per_task_loss: {'cls/loss': 1.23, 'seg/loss': 0.45, ...}
            total_loss: 总 loss.
            step: 当前 step.
        """
        self.log_scalar("loss/total", total_loss, step)
        for tag, value in per_task_loss.items():
            self.log_scalar(f"loss/{tag.replace('/', '_')}", value, step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """记录评测指标.

        Args:
            metrics: {'cls/acc': 0.85, 'cls/mAP': 0.82, ...}
            step: 当前 step.
        """
        for tag, value in metrics.items():
            self.log_scalar(f"metrics/{tag.replace('/', '_')}", value, step)

    def log_lr(self, lr: float, step: int) -> None:
        """记录学习率."""
        self.log_scalar("train/lr", lr, step)

    # ---- 直方图 (可选) ----

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """记录张量值分布."""
        if self.enabled and self._writer is not None:
            self._writer.add_histogram(tag, values, step)

    def log_gradients(self, model: nn.Module, step: int) -> None:
        """记录梯度范数分布 (谨慎使用, 有性能开销)."""
        if not self.enabled or self._writer is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._writer.add_histogram(f"gradients/{name}", param.grad, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """记录文本 (超参 / 配置)."""
        if self.enabled and self._writer is not None:
            self._writer.add_text(tag, text, step)

    def flush(self) -> None:
        """强制写入磁盘."""
        if self.enabled and self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """关闭 TensorBoard writer."""
        if self.enabled and self._writer is not None:
            self._writer.close()
            self._writer = None


# ══════════════════════════════════════════════════════════════════════
# CSV Logger
# ══════════════════════════════════════════════════════════════════════


class CSVLogger:
    """将训练/评测指标追加写入 logs/results.csv.

    自动管理表头: 首次写入时根据 dict keys 生成 CSV header。
    追加模式下不会覆盖已有数据。

    Args:
        log_dir: CSV 文件目录.
        filename: CSV 文件名 (默认 results.csv).
        resume: 是否续写 (True=追加, False=覆盖).
    """

    HEADER = [
        "step", "epoch", "timestamp",
        "cls/loss", "seg/loss", "det/loss", "cnt/loss", "total_loss",
        "cls/acc", "cls/mAP", "cls/BA",
        "seg/mIoU", "seg/mAcc",
        "det/AP50", "det/AP",
        "cnt/MAE", "cnt/RMSE", "cnt/R2",
        "aggregate", "lr",
    ]

    def __init__(self, log_dir: str, filename: str = "results.csv", resume: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.resume = resume

        # 如果新文件或覆盖模式, 写入表头
        if not self.filepath.exists() or not resume:
            self._write_header()

    def _write_header(self) -> None:
        """写入 CSV 表头."""
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)

    def log(self, row: Dict[str, Any]) -> None:
        """追加一行数据.

        Args:
            row: 包含 step/epoch/metrics 等任意字段的 dict.
                 缺失字段自动填空字符串。
        """
        # 自动添加时间戳
        if "timestamp" not in row:
            row["timestamp"] = datetime.now().isoformat()

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(col, "") for col in self.HEADER])

    def log_step(
        self,
        step: int,
        epoch: int,
        losses: Optional[Dict[str, float]] = None,
        total_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        """便捷方法: 记录一个训练 step 的完整信息.

        Args:
            step: 全局 step 计数.
            epoch: 当前 epoch.
            losses: 各任务 loss dict.
            total_loss: 总 loss.
            metrics: 评测指标 dict.
            lr: 当前学习率.
        """
        row: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
        }

        if losses:
            for k, v in losses.items():
                row[f"{k}"] = v
        if total_loss is not None:
            row["total_loss"] = total_loss
        if metrics:
            for k, v in metrics.items():
                # 适配 cls/acc → cls/acc
                row[f"{k}"] = v
        if lr is not None:
            row["lr"] = lr

        self.log(row)

    def load_all(self) -> list[Dict[str, str]]:
        """读取 CSV 全部行."""
        if not self.filepath.exists():
            return []
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_last_step(self) -> int:
        """获取最后记录的 step 数 (用于 resume)."""
        rows = self.load_all()
        if rows:
            try:
                return int(rows[-1].get("step", -1))
            except (ValueError, KeyError):
                return -1
        return -1


# ══════════════════════════════════════════════════════════════════════
# Checkpoint Manager
# ══════════════════════════════════════════════════════════════════════


class CheckpointManager:
    """Checkpoint 管理 — 保存 / 加载 / 最优追踪.

    文件命名 (api_contract.md §9):
      - ckpt_step{step}.pt: 按 step 定期保存
      - best.pt:            历史最优指标
      - last.pt:            最近一次保存 (resume 用)

    Checkpoint 内容:
      {
          'step': int,
          'epoch': int,
          'model_state': OrderedDict,
          'optimizer_state': dict,
          'scheduler_state': dict | None,
          'loss_agg_state': dict | None,
          'best_metric': dict,
          'cfg': dict,
      }

    Args:
        out_dir: checkpoint 输出目录.
        save_every: 每隔多少 step 保存一次 ckpt_step{N}.pt.
        keep_last_n: 保留最近 N 个 step checkpoint (旧自动删除).
        mode: 'max' (指标越大越好, 如 acc/mAP) 或 'min' (越小越好, 如 loss).
    """

    def __init__(
        self,
        out_dir: str,
        save_every: int = 5000,
        keep_last_n: int = 3,
        mode: str = "max",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.mode = mode

        # 最优追踪
        self.best_metric_value: float = -float("inf") if mode == "max" else float("inf")
        self.best_step: int = 0

        # 历史 step ckpt 记录 (用于清理)
        self._saved_steps: list[int] = []

    # ---- 保存 ----

    def save(
        self,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_agg: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, float]] = None,
        cfg: Optional[Dict] = None,
    ) -> str:
        """保存完整 checkpoint.

        Args:
            step: 全局 step.
            epoch: 当前 epoch.
            model: MTLModel 实例.
            optimizer: 优化器.
            scheduler: 学习率调度器 (可选).
            loss_agg: LossAggregator 实例 (可选).
            metrics: 当前评测指标 (可选).
            cfg: 配置 dict (可选).

        Returns:
            保存的文件路径.
        """
        ckpt: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "loss_agg_state": loss_agg.state_dict() if loss_agg is not None else None,
            "best_metric": {
                "value": self.best_metric_value,
                "step": self.best_step,
            },
            "cfg": cfg,
        }

        # 追加当前 metric 快照
        if metrics:
            ckpt["metrics"] = metrics

        filepath = self.out_dir / f"ckpt_step{step}.pt"
        torch.save(ckpt, filepath)

        # 更新 last.pt
        last_path = self.out_dir / "last.pt"
        shutil.copy2(filepath, last_path)

        # 追踪 & 清理
        self._saved_steps.append(step)
        self._cleanup_old()

        return str(filepath)

    def save_best(
        self,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_value: float,
        scheduler: Optional[Any] = None,
        loss_agg: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, float]] = None,
        cfg: Optional[Dict] = None,
    ) -> Optional[str]:
        """如果当前指标优于历史最优, 保存 best.pt.

        Args:
            step: 全局 step.
            epoch: 当前 epoch.
            model: MTLModel 实例.
            optimizer: 优化器.
            metric_value: 用于比较的单个指标值 (如 aggregate / cls/mAP).
            scheduler: 学习率调度器.
            loss_agg: LossAggregator.
            metrics: 完整指标 dict.
            cfg: 配置 dict.

        Returns:
            保存路径 (如果更新了 best), 否则 None.
        """
        is_better = (
            (self.mode == "max" and metric_value > self.best_metric_value)
            or (self.mode == "min" and metric_value < self.best_metric_value)
        )

        if not is_better:
            return None

        self.best_metric_value = metric_value
        self.best_step = step

        ckpt: Dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "loss_agg_state": loss_agg.state_dict() if loss_agg is not None else None,
            "best_metric": {
                "value": self.best_metric_value,
                "step": self.best_step,
            },
            "cfg": cfg,
        }
        if metrics:
            ckpt["metrics"] = metrics

        filepath = self.out_dir / "best.pt"
        torch.save(ckpt, filepath)
        return str(filepath)

    # ---- 加载 ----

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """加载 checkpoint.

        Args:
            path: checkpoint 路径; 为 None 时自动加载 last.pt.

        Returns:
            Checkpoint dict.

        Raises:
            FileNotFoundError: 指定路径不存在.
        """
        if path is None:
            path = str(self.out_dir / "last.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu")

    def load_last(self) -> Dict[str, Any]:
        """加载 last.pt."""
        return self.load(str(self.out_dir / "last.pt"))

    def load_best(self) -> Dict[str, Any]:
        """加载 best.pt."""
        return self.load(str(self.out_dir / "best.pt"))

    # ---- 内部 ----

    def _cleanup_old(self) -> None:
        """删除超出 keep_last_n 数量的旧 checkpoint."""
        while len(self._saved_steps) > self.keep_last_n:
            old_step = self._saved_steps.pop(0)
            old_path = self.out_dir / f"ckpt_step{old_step}.pt"
            if old_path.exists():
                old_path.unlink()

    # ---- 断点续训信息 ----

    def get_resume_info(self) -> Dict[str, Any]:
        """获取用于 resume 的信息.

        Returns:
            {'step': int, 'epoch': int, 'best_metric_value': float}
            如果 last.pt 不存在则返回初始值。
        """
        try:
            ckpt = self.load_last()
            self.best_metric_value = ckpt.get("best_metric", {}).get("value", self.best_metric_value)
            self.best_step = ckpt.get("best_metric", {}).get("step", 0)
            return {
                "step": ckpt["step"],
                "epoch": ckpt["epoch"],
                "best_metric_value": self.best_metric_value,
            }
        except FileNotFoundError:
            return {"step": 0, "epoch": 0, "best_metric_value": self.best_metric_value}


# ══════════════════════════════════════════════════════════════════════
# 便捷工具
# ══════════════════════════════════════════════════════════════════════


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """格式化指标 dict 为单行字符串 (用于终端打印).

    Example:
        'cls/acc=0.8500 cls/mAP=0.8200 cls/BA=0.8000'
    """
    parts = [f"{k}={v:.{precision}f}" for k, v in sorted(metrics.items())]
    return " ".join(parts)


def format_losses(losses: Dict[str, float], precision: int = 3) -> str:
    """格式化 loss dict 为单行字符串.

    Example:
        'cls/loss=1.234 seg/loss=0.456 det/loss=0.789 cnt/loss=2.345'
    """
    parts = [f"{k}={v:.{precision}f}" for k, v in sorted(losses.items())]
    return " ".join(parts)
