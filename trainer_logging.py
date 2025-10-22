# -*- coding: utf-8 -*-
"""Utility classes for structured training logs."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Optional


def _format_ms(seconds: float) -> str:
    """Format elapsed seconds as milliseconds string."""
    return f"{seconds * 1000:.2f} ms"


class TrainingLogger:
    """Console + CSV logger shared by long-tail and open-set training flows."""

    def __init__(
        self,
        log_file: str,
        console_interval: int = 1,
        extra_columns: Optional[Dict[str, str]] = None,
    ) -> None:
        self.log_file = log_file
        self.console_interval = console_interval
        self.epoch_start_time: Optional[float] = None
        self.per_epoch_times = []
        self.extra_column_labels: Dict[str, str] = extra_columns or {}
        self.extra_columns = list(self.extra_column_labels.keys())

        header = (
            "epoch,timestamp,train_loss,train_acc,val_loss,val_acc,val_balanced_acc,"
            "majority_acc,medium_acc,minority_acc,lr,epoch_time"
        )
        if self.extra_columns:
            header += "," + ",".join(self.extra_columns)

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header + "\n")

    def start_epoch(self, epoch: int, lr: Optional[float] = None) -> None:
        """Mark the start of an epoch and optionally print a banner."""
        self.epoch_start_time = time.time()
        if epoch % self.console_interval == 0:
            lr_info = f" | LR: {lr:.2e}" if lr is not None else ""
            print(f"\n{'=' * 80}\nEpoch {epoch}{lr_info}\n{'=' * 80}")

    def log_training_step(
        self,
        batch_idx: int,
        total_batches: int,
        loss: float,
        acc: float,
        lr: float,
    ) -> None:
        """Emit periodic training step updates."""
        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / max(total_batches, 1) * 100
            print(
                f"  [Train] Batch {batch_idx + 1:4d}/{total_batches} ({progress:5.1f}%) | "
                f"Loss: {loss:.6f} | Acc: {acc:5.2f}% | LR: {lr:.2e}"
            )

    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        group_metrics: Dict[str, Dict[str, float]],
        lr: float,
        *,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record epoch-level metrics and print a summary."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if epoch % self.console_interval == 0:
            print(f"\n{'─' * 80}")
            print(f"Epoch {epoch} Summary | Learning Rate: {lr:.2e}")
            print(f"  Time: {_format_ms(epoch_time)}")
            print(
                f"  Train - Loss: {train_metrics.get('loss', 0.0):.6f} | "
                f"Acc: {train_metrics.get('acc', 0.0):5.2f}%"
            )
            print(
                f"  Val   - Loss: {val_metrics.get('loss', 0.0):.6f} | "
                f"Acc: {val_metrics.get('acc', 0.0):5.2f}% | "
                f"Bal Acc: {val_metrics.get('balanced_acc', 0.0):5.2f}%"
            )
            if group_metrics:
                print("  Class Groups:")
                for group_name, metrics in group_metrics.items():
                    if not metrics:
                        continue
                    accuracy = metrics.get("accuracy", 0.0)
                    f1 = metrics.get("f1", 0.0)
                    support = int(metrics.get("support", 0))
                    print(
                        f"    {group_name.capitalize():8s}: Acc={accuracy:5.2f}% | "
                        f"F1={f1:5.2f}% | Support={support:4d}"
                    )
            if self.extra_columns and extra_metrics:
                print("  Open-Set Metrics:")
                for key in self.extra_columns:
                    value = extra_metrics.get(key)
                    if value is None:
                        continue
                    label = self.extra_column_labels.get(key, key)
                    print(f"    {label}: {value:5.2f}%")

        maj = group_metrics.get("majority", {}).get("accuracy", 0.0)
        med = group_metrics.get("medium", {}).get("accuracy", 0.0)
        mino = group_metrics.get("minority", {}).get("accuracy", 0.0)
        row = (
            f"{epoch},{ts},{train_metrics.get('loss', 0.0):.6f},{train_metrics.get('acc', 0.0):.4f},"
            f"{val_metrics.get('loss', 0.0):.6f},{val_metrics.get('acc', 0.0):.4f},{val_metrics.get('balanced_acc', 0.0):.4f},"
            f"{maj:.4f},{med:.4f},{mino:.4f},{lr:.8f},{epoch_time:.2f}"
        )
        if self.extra_columns:
            metrics = extra_metrics or {}
            for key in self.extra_columns:
                row += f",{metrics.get(key, 0.0):.4f}"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(row + "\n")

        self.per_epoch_times.append(epoch_time)
