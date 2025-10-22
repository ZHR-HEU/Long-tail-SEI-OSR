# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from datetime import datetime
def _format_ms(seconds):
    return f"{seconds * 1000:.2f} ms"
class TrainingLogger:
    def __init__(self, log_file: str, console_interval: int = 1):
        self.log_file = log_file
        self.console_interval = console_interval
        self.epoch_start_time = None
        self.per_epoch_times = []
        with open(log_file, 'w') as f:
            f.write("epoch,timestamp,train_loss,train_acc,val_loss,val_acc,val_balanced_acc,"
                    "majority_acc,medium_acc,minority_acc,lr,epoch_time\n")

    def start_epoch(self, epoch: int, lr: float = None):  # 添加lr参数
        self.epoch_start_time = time.time()
        if epoch % self.console_interval == 0:
            lr_info = f" | LR: {lr:.2e}" if lr is not None else ""
            print(f"\n{'='*80}\nEpoch {epoch}{lr_info}\n{'='*80}")

    def log_training_step(self, batch_idx: int, total_batches: int, loss: float, acc: float, lr: float):
        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  [Train] Batch {batch_idx+1:4d}/{total_batches} ({progress:5.1f}%) | "
                  f"Loss: {loss:.6f} | Acc: {acc:5.2f}% | LR: {lr:.2e}")

    def log_epoch_end(self, epoch: int, train_metrics, val_metrics, group_metrics, lr: float):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if epoch % self.console_interval == 0:
            print(f"\n{'─'*80}")  # 添加分隔线使输出更清晰
            print(f"Epoch {epoch} Summary | Learning Rate: {lr:.2e}")  # 突出显示LR
            print(f"  Time: {_format_ms(epoch_time)}")
            print(f"  Train - Loss: {train_metrics['loss']:.6f} | Acc: {train_metrics['acc']:5.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f} | Acc: {val_metrics['acc']:5.2f}% | "
                  f"Bal Acc: {val_metrics['balanced_acc']:5.2f}%")
            if group_metrics:
                print("  Class Groups:")
                for g, m in group_metrics.items():
                    if m:
                        print(f"    {g.capitalize():8s}: Acc={m['accuracy']:5.2f}% | F1={m['f1']:5.2f}% | Support={m['support']:4d}")

        with open(self.log_file, 'a') as f:
            maj = group_metrics.get('majority', {}).get('accuracy', 0)
            med = group_metrics.get('medium', {}).get('accuracy', 0)
            mino= group_metrics.get('minority', {}).get('accuracy', 0)
            f.write(f"{epoch},{ts},{train_metrics['loss']:.6f},{train_metrics['acc']:.4f},"
                    f"{val_metrics['loss']:.6f},{val_metrics['acc']:.4f},{val_metrics['balanced_acc']:.4f},"
                    f"{maj:.4f},{med:.4f},{mino:.4f},{lr:.8f},{epoch_time:.2f}\n")
        self.per_epoch_times.append(epoch_time)
