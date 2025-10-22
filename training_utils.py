"""
Complete collection of training strategies for imbalanced learning.

This module contains various training strategies including learning rate scheduling,
early stopping, model checkpointing, and other training utilities commonly used
in imbalanced learning scenarios.

All dependencies are included - no external imports needed beyond PyTorch.

Author: Consolidated from multiple implementations
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import os
import time
from collections import defaultdict


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class GradualWarmupScheduler(LRScheduler):
    """
    Gradually warm-up (increasing) learning rate in optimizer.

    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    This scheduler gradually increases the learning rate from 0 to the target
    learning rate over a specified number of epochs, then optionally continues
    with another scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier (float): Target learning rate = base lr * multiplier if multiplier > 1.0.
                           If multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch (int): Target learning rate is reached at total_epoch, gradually.
        after_scheduler: After target_epoch, use this scheduler (eg. ReduceLROnPlateau).
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """Get current learning rates."""
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """Special step method for ReduceLROnPlateau scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                         for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """Step method supporting both regular schedulers and ReduceLROnPlateau."""
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class CosineAnnealingWarmupRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts and initial warmup.

    Combines warmup with cosine annealing restart for better convergence
    in imbalanced learning scenarios.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): Number of steps in the first cycle.
        cycle_mult (float): Cycle length multiplier after each restart.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        gamma (float): Learning rate decay factor after each restart.
    """

    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=1e-3, min_lr=1e-4,
                 warmup_steps=0, gamma=1.0):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer)

        # Set initial learning rate
        self.init_lr()

    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        """Calculate current learning rates."""
        if self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cos_steps = self.step_in_cycle - self.warmup_steps
            cos_total = self.cur_cycle_steps - self.warmup_steps
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + np.cos(np.pi * cos_steps / cos_total)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.step_in_cycle == self.cur_cycle_steps:
            # Restart cycle
            self.cycle += 1
            self.step_in_cycle = 0
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
            self.max_lr *= self.gamma

        self.step_in_cycle += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# =============================================================================
# Early Stopping and Model Checkpointing
# =============================================================================

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This implementation supports multiple metrics monitoring and flexible checkpoint saving.

    Args:
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
        trace_func (function): Trace print function.
        save_best_only (bool): Only save the best model.
        monitor (str): Metric to monitor ('loss' or 'acc').
        mode (str): 'min' for loss, 'max' for accuracy.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',
                 trace_func=print, save_best_only=True, monitor='loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        if mode == 'min':
            self.val_best = np.inf
            self.is_better = lambda current, best: current < best - self.delta
        else:  # mode == 'max'
            self.val_best = -np.inf
            self.is_better = lambda current, best: current > best + self.delta

    def __call__(self, val_metric, model, optimizer=None, epoch=None, **kwargs):
        """
        Check if early stopping criteria is met.

        Args:
            val_metric (float): Current validation metric value.
            model (nn.Module): Model to potentially save.
            optimizer: Optimizer state to save.
            epoch (int): Current epoch number.
            **kwargs: Additional information to save in checkpoint.
        """
        current = val_metric

        if self.best_score is None:
            self.best_score = current
            self.save_checkpoint(current, model, optimizer, epoch, **kwargs)
        elif not self.is_better(current, self.best_score):
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.trace_func(f'Current {self.monitor}: {current:.6f}, Best {self.monitor}: {self.best_score:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current
            self.save_checkpoint(current, model, optimizer, epoch, **kwargs)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, optimizer=None, epoch=None, **kwargs):
        """Save model when validation metric improves."""
        if self.verbose:
            direction = 'decreased' if self.mode == 'min' else 'increased'
            self.trace_func(
                f'Validation {self.monitor} {direction} ({self.val_best:.6f} --> {val_metric:.6f}). Saving model ...')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_metric': val_metric,
            'epoch': epoch,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add any additional kwargs
        checkpoint.update(kwargs)

        torch.save(checkpoint, self.path)
        self.val_best = val_metric


class ModelCheckpointer:
    """
    Advanced model checkpointing with multiple save strategies.

    Supports saving best models, regular intervals, and custom conditions.

    Args:
        save_dir (str): Directory to save checkpoints.
        save_best (bool): Whether to save the best model.
        save_interval (int): Save every N epochs (0 to disable).
        max_checkpoints (int): Maximum number of checkpoints to keep.
        monitor (str): Metric to monitor for best model.
        mode (str): 'min' or 'max' for the monitored metric.
    """

    def __init__(self, save_dir='checkpoints', save_best=True, save_interval=0,
                 max_checkpoints=5, monitor='loss', mode='min'):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Track best metric
        if mode == 'min':
            self.best_metric = np.inf
            self.is_better = lambda current, best: current < best
        else:
            self.best_metric = -np.inf
            self.is_better = lambda current, best: current > best

        # Track saved checkpoints
        self.saved_checkpoints = []

    def save(self, model, optimizer=None, epoch=None, metrics=None, **kwargs):
        """
        Save checkpoint based on configured strategy.

        Args:
            model (nn.Module): Model to save.
            optimizer: Optimizer to save.
            epoch (int): Current epoch.
            metrics (dict): Dictionary of metrics.
            **kwargs: Additional data to save.
        """
        if metrics is None:
            metrics = {}

        # Check if we should save based on best metric
        save_as_best = False
        if self.save_best and self.monitor in metrics:
            current_metric = metrics[self.monitor]
            if self.is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
                save_as_best = True

        # Check if we should save based on interval
        save_interval = (self.save_interval > 0 and
                         epoch is not None and
                         epoch % self.save_interval == 0)

        # Save if any condition is met
        if save_as_best or save_interval:
            timestamp = int(time.time())

            if save_as_best:
                filename = f'best_model_epoch_{epoch}_{timestamp}.pth'
            else:
                filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'

            filepath = os.path.join(self.save_dir, filename)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': timestamp,
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            checkpoint.update(kwargs)

            torch.save(checkpoint, filepath)
            self.saved_checkpoints.append(filepath)

            # Remove old checkpoints if necessary
            if len(self.saved_checkpoints) > self.max_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

            print(f"Checkpoint saved: {filepath}")


# =============================================================================
# Training Utilities
# =============================================================================

class MetricsTracker:
    """
    Track and manage training metrics with history and statistics.

    Supports metric averaging, best value tracking, and history management.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            self.current_metrics[key] = value

            # Update best metrics
            if key not in self.best_metrics:
                self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}
            else:
                # For loss metrics (assume lower is better if 'loss' in name)
                if 'loss' in key.lower():
                    if value < self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}
                # For other metrics (assume higher is better)
                else:
                    if value > self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': len(self.metrics[key]) - 1}

    def get_average(self, metric_name, last_n=None):
        """Get average of a metric over last N epochs."""
        if metric_name not in self.metrics:
            return None

        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]

        return np.mean(values) if values else None

    def get_best(self, metric_name):
        """Get best value and epoch for a metric."""
        return self.best_metrics.get(metric_name, None)

    def get_current(self, metric_name):
        """Get current value of a metric."""
        return self.current_metrics.get(metric_name, None)

    def get_history(self, metric_name):
        """Get full history of a metric."""
        return self.metrics.get(metric_name, [])

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.best_metrics.clear()
        self.current_metrics.clear()

    def summary(self):
        """Get summary of all metrics."""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = {
                'current': self.get_current(metric_name),
                'best': self.get_best(metric_name),
                'average_last_5': self.get_average(metric_name, 5),
                'total_epochs': len(self.metrics[metric_name])
            }
        return summary


class TrainingManager:
    """
    Comprehensive training manager that combines all training strategies.

    Integrates scheduler, early stopping, checkpointing, and metrics tracking.

    Args:
        model (nn.Module): Model to train.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        early_stopping (EarlyStopping): Early stopping configuration.
        checkpointer (ModelCheckpointer): Checkpoint manager.
        metrics_tracker (MetricsTracker): Metrics tracking.
    """

    def __init__(self, model, optimizer, scheduler=None, early_stopping=None,
                 checkpointer=None, metrics_tracker=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpointer = checkpointer
        self.metrics_tracker = metrics_tracker or MetricsTracker()

        self.epoch = 0
        self.should_stop = False

    def step_epoch(self, metrics=None):
        """
        Perform end-of-epoch operations.

        Args:
            metrics (dict): Metrics for the current epoch.
        """
        if metrics is None:
            metrics = {}

        self.epoch += 1

        # Update metrics tracker
        self.metrics_tracker.update(**metrics)

        # Step scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, GradualWarmupScheduler) and hasattr(self.scheduler,
                                                                              'after_scheduler') and isinstance(
                    self.scheduler.after_scheduler, ReduceLROnPlateau):
                # Special handling for warmup + ReduceLROnPlateau
                monitor_metric = metrics.get('val_loss', metrics.get('loss', 0))
                self.scheduler.step_ReduceLROnPlateau(monitor_metric, self.epoch)
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                monitor_metric = metrics.get('val_loss', metrics.get('loss', 0))
                self.scheduler.step(monitor_metric)
            else:
                self.scheduler.step()

        # Check early stopping
        if self.early_stopping is not None:
            monitor_metric = metrics.get(f'val_{self.early_stopping.monitor}',
                                         metrics.get(self.early_stopping.monitor, 0))
            self.early_stopping(monitor_metric, self.model, self.optimizer, self.epoch, **metrics)
            self.should_stop = self.early_stopping.early_stop

        # Save checkpoint
        if self.checkpointer is not None:
            self.checkpointer.save(self.model, self.optimizer, self.epoch, metrics)

    def get_current_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def should_stop_training(self):
        """Check if training should stop."""
        return self.should_stop

    def get_metrics_summary(self):
        """Get summary of all metrics."""
        return self.metrics_tracker.summary()


# =============================================================================
# Factory Functions
# =============================================================================

def create_warmup_scheduler(optimizer, warmup_epochs, after_scheduler=None, multiplier=1.0):
    """
    Create a warmup scheduler with optional after scheduler.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs (int): Number of warmup epochs.
        after_scheduler: Scheduler to use after warmup.
        multiplier (float): Learning rate multiplier.

    Returns:
        GradualWarmupScheduler: Configured warmup scheduler.
    """
    return GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=multiplier,
        total_epoch=warmup_epochs,
        after_scheduler=after_scheduler
    )


def create_early_stopping(patience=7, monitor='loss', mode='min', save_path='best_model.pth', verbose=True):
    """
    Create an early stopping callback.

    Args:
        patience (int): Number of epochs to wait.
        monitor (str): Metric to monitor.
        mode (str): 'min' or 'max'.
        save_path (str): Path to save best model.
        verbose (bool): Whether to print messages.

    Returns:
        EarlyStopping: Configured early stopping.
    """
    return EarlyStopping(
        patience=patience,
        monitor=monitor,
        mode=mode,
        path=save_path,
        verbose=verbose
    )


def create_training_manager(model, optimizer, warmup_epochs=5, patience=10,
                            save_dir='checkpoints', monitor='loss', after_scheduler=None):
    """
    Create a complete training manager with common configurations.

    Args:
        model (nn.Module): Model to train.
        optimizer: PyTorch optimizer.
        warmup_epochs (int): Number of warmup epochs.
        patience (int): Early stopping patience.
        save_dir (str): Directory for checkpoints.
        monitor (str): Metric to monitor.
        after_scheduler: Scheduler to use after warmup.

    Returns:
        TrainingManager: Configured training manager.
    """
    # Create components
    scheduler = create_warmup_scheduler(optimizer, warmup_epochs, after_scheduler)
    early_stopping = create_early_stopping(patience=patience, monitor=monitor)
    checkpointer = ModelCheckpointer(save_dir=save_dir, monitor=monitor)
    metrics_tracker = MetricsTracker()

    return TrainingManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpointer=checkpointer,
        metrics_tracker=metrics_tracker
    )


# =============================================================================
# Usage Examples
# =============================================================================

"""
Usage Examples:

1. Basic warmup scheduler:
   from torch.optim.lr_scheduler import StepLR
   from torch.optim import SGD

   optimizer = SGD(model.parameters(), lr=0.1)
   step_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
   warmup_scheduler = create_warmup_scheduler(optimizer, warmup_epochs=5, 
                                            after_scheduler=step_scheduler)

   # In training loop:
   for epoch in range(epochs):
       # ... training code ...
       warmup_scheduler.step()

2. Early stopping:
   early_stopping = create_early_stopping(patience=10, monitor='val_loss', 
                                         save_path='best_model.pth')

   # In training loop:
   for epoch in range(epochs):
       # ... training and validation ...
       early_stopping(val_loss, model, optimizer, epoch)
       if early_stopping.early_stop:
           break

3. Complete training manager:
   manager = create_training_manager(
       model=model,
       optimizer=optimizer,
       warmup_epochs=5,
       patience=10,
       save_dir='./checkpoints'
   )

   # In training loop:
   for epoch in range(epochs):
       # ... training code ...
       train_loss = train_one_epoch(model, train_loader, optimizer)
       val_loss = validate(model, val_loader)

       manager.step_epoch({'train_loss': train_loss, 'val_loss': val_loss})

       if manager.should_stop_training():
           print("Early stopping triggered")
           break

   # Get training summary
   summary = manager.get_metrics_summary()
   print(summary)

4. Cosine annealing with warmup:
   scheduler = CosineAnnealingWarmupRestarts(
       optimizer, 
       first_cycle_steps=50,
       cycle_mult=2.0,
       max_lr=1e-3,
       min_lr=1e-5,
       warmup_steps=5
   )

5. Manual metrics tracking:
   tracker = MetricsTracker()

   for epoch in range(epochs):
       # ... training ...
       tracker.update(train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)

       # Get best validation accuracy
       best_acc = tracker.get_best('val_acc')
       print(f"Best val_acc: {best_acc['value']:.4f} at epoch {best_acc['epoch']}")
"""