# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, List
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from omegaconf import DictConfig
from training_utils import create_warmup_scheduler, CosineAnnealingWarmupRestarts
from imbalanced_losses import create_loss as get_loss_function


def build_scheduler_for_stage(optimizer, cfg: DictConfig, epochs_this_stage: int, stage: str = 'stage1'):
    def pick(name: str):
        if stage == 'stage2' and hasattr(cfg.stage2, name) and getattr(cfg.stage2, name) is not None:
            return getattr(cfg.stage2, name)
        return getattr(cfg.scheduler, name)

    scheduler_name = pick('name') if stage == 'stage2' else cfg.scheduler.name
    after_scheduler = None
    scheduler = None

    if scheduler_name == 'cosine':
        after_scheduler = CosineAnnealingLR(optimizer, T_max=epochs_this_stage, eta_min=1e-6)
    elif scheduler_name == 'step':
        after_scheduler = StepLR(optimizer, step_size=pick('step_size'), gamma=pick('gamma'))
    elif scheduler_name == 'plateau':
        plateau_mode = 'min' if 'loss' in cfg.early_stopping.monitor else 'max'
        after_scheduler = ReduceLROnPlateau(
            optimizer, mode=plateau_mode,
            factor=pick('plateau_factor'), patience=pick('plateau_patience'),
            min_lr=pick('plateau_min_lr'))
    elif scheduler_name == 'cosine_warmup_restarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=pick('cos_first_cycle_steps'),
            cycle_mult=pick('cos_cycle_mult'),
            max_lr=pick('cos_max_lr') if pick('cos_max_lr') is not None else optimizer.param_groups[0]['lr'],
            min_lr=pick('cos_min_lr'),
            warmup_steps=pick('cos_warmup_steps'),
            gamma=pick('cos_gamma')
        )
    else:
        scheduler = None

    if scheduler_name != 'cosine_warmup_restarts':
        warmup_epochs = pick('warmup_epochs')
        warmup_mult = pick('warmup_multiplier')
        if after_scheduler is not None and warmup_epochs and warmup_epochs > 0:
            scheduler = create_warmup_scheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                after_scheduler=after_scheduler,
                multiplier=warmup_mult if warmup_mult is not None else 1.0
            )
        else:
            scheduler = after_scheduler
    return scheduler


def build_optimizer(optimizer_name: str, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float):
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_optimizer_with_groups(optimizer_name: str, param_groups: List[dict], weight_decay: float):
    if optimizer_name == 'SGD':
        return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == 'Adam':
        return optim.Adam(param_groups, weight_decay=weight_decay)
    if optimizer_name == 'AdamW':
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_criterion(loss_name: str, cfg: DictConfig, class_counts):
    """
    构建损失函数

    Args:
        loss_name: 损失函数名称
        cfg: 配置对象
        class_counts: 类别样本数统计

    Returns:
        损失函数实例
    """
    loss_kwargs = {}

    # ===================================================================
    # FocalLoss 参数
    # ===================================================================
    if loss_name == 'FocalLoss':
        loss_kwargs['gamma'] = getattr(cfg.loss, 'focal_gamma', 2.0)
        if hasattr(cfg.loss, 'focal_alpha') and cfg.loss.focal_alpha is not None:
            loss_kwargs['alpha'] = cfg.loss.focal_alpha

    # ===================================================================
    # CrossEntropy 参数
    # ===================================================================
    if loss_name == 'CrossEntropy':
        if cfg.training.label_smoothing > 0:
            loss_kwargs['label_smoothing'] = cfg.training.label_smoothing
        if hasattr(cfg.loss, 'weight') and cfg.loss.weight is not None:
            loss_kwargs['weight'] = cfg.loss.weight

    # ===================================================================
    # ClassBalancedLoss 参数
    # ===================================================================
    if loss_name == 'ClassBalancedLoss':
        loss_kwargs['beta'] = getattr(cfg.loss, 'cb_beta', 0.9999)

    # ===================================================================
    # LDAMLoss 参数
    # ===================================================================
    if loss_name == 'LDAMLoss':
        loss_kwargs['max_margin'] = getattr(cfg.loss, 'ldam_max_margin', 0.5)
        loss_kwargs['scale'] = getattr(cfg.loss, 'ldam_scale', 30.0)
        loss_kwargs['drw_start_epoch'] = getattr(cfg.loss, 'ldam_drw_start', 0)

    # ===================================================================
    # LogitAdjustmentLoss 参数
    # ===================================================================
    if loss_name == 'LogitAdjustmentLoss':
        loss_kwargs['tau'] = getattr(cfg.loss, 'logit_tau', 1.0)

    # ===================================================================
    # BalancedSoftmaxLoss 无额外参数
    # ===================================================================

    # ===================================================================
    # ProgressiveLoss 参数
    # ===================================================================
    if loss_name == 'ProgressiveLoss':
        loss_kwargs['total_epochs'] = cfg.training.epochs
        loss_kwargs['start_strategy'] = getattr(cfg.loss, 'prog_start_strategy', 'uniform')
        loss_kwargs['end_strategy'] = getattr(cfg.loss, 'prog_end_strategy', 'inverse')

    # ===================================================================
    # Cost-Sensitive Losses 参数（重要：这里不能嵌套在上面的 if 内）
    # ===================================================================
    if loss_name in ['CostSensitiveCE', 'CostSensitiveExpected', 'CostSensitiveFocal']:
        # 优先从 stage2 配置读取，其次从 loss 配置
        stage2_cfg = getattr(cfg, 'stage2', None)
        loss_cfg = getattr(cfg, 'loss', None)

        # 策略选择
        cost_strategy = 'auto'  # 默认自动生成
        if stage2_cfg and hasattr(stage2_cfg, 'cost_strategy'):
            cost_strategy = stage2_cfg.cost_strategy
        elif loss_cfg and hasattr(loss_cfg, 'cost_strategy'):
            cost_strategy = loss_cfg.cost_strategy

        loss_kwargs['cost_strategy'] = cost_strategy

        # 手动模式需要提供向量或矩阵
        if cost_strategy == 'manual':
            if stage2_cfg and hasattr(stage2_cfg, 'cost_vector'):
                loss_kwargs['cost_vector'] = stage2_cfg.cost_vector
            elif stage2_cfg and hasattr(stage2_cfg, 'cost_matrix'):
                loss_kwargs['cost_matrix'] = stage2_cfg.cost_matrix
            elif loss_cfg and hasattr(loss_cfg, 'cost_vector'):
                loss_kwargs['cost_vector'] = loss_cfg.cost_vector
            elif loss_cfg and hasattr(loss_cfg, 'cost_matrix'):
                loss_kwargs['cost_matrix'] = loss_cfg.cost_matrix

        # 自动模式需要 class_counts
        if cost_strategy in ['auto', 'sqrt', 'log']:
            loss_kwargs['class_counts'] = class_counts

        # CostSensitiveFocal 特殊参数
        if loss_name == 'CostSensitiveFocal':
            gamma = 2.0
            if stage2_cfg and hasattr(stage2_cfg, 'focal_gamma'):
                gamma = stage2_cfg.focal_gamma
            elif loss_cfg and hasattr(loss_cfg, 'focal_gamma'):
                gamma = loss_cfg.focal_gamma
            loss_kwargs['gamma'] = gamma

    # ===================================================================
    # 需要 class_counts 的损失函数（统一处理）
    # ===================================================================
    LOSSES_NEED_COUNTS = {
        'ClassBalancedLoss',
        'LDAMLoss',
        'BalancedSoftmaxLoss',
        'LogitAdjustmentLoss',
        'ProgressiveLoss',
    }

    if loss_name in LOSSES_NEED_COUNTS:
        loss_kwargs['class_counts'] = class_counts

    # ===================================================================
    # 创建并返回损失函数（关键：必须在函数末尾）
    # ===================================================================
    try:
        criterion = get_loss_function(loss_name, **loss_kwargs)
        if criterion is None:
            raise ValueError(f"get_loss_function returned None for {loss_name}")
        print(f"[Criterion] Successfully created {loss_name}")
        return criterion
    except Exception as e:
        print(f"[ERROR] Failed to create loss function '{loss_name}'")
        print(f"  Error: {e}")
        print(f"  loss_kwargs: {loss_kwargs}")
        raise