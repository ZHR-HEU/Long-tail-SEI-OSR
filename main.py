#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Training Script for Imbalanced Learning

核心功能：
- 数据加载与采样
- Loss函数与优化器
- Warmup调度器
- 两阶段训练
- 简化的日志和检查点管理
- 可视化功能
- 时间跟踪统计
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

# 项目模块
from data_utils import LoaderConfig, build_dataloaders, make_long_tailed_indices, ADSBSignalDataset
from imbalanced_losses import create_loss
from models import create_model
from optim_utils import build_scheduler_for_stage, build_optimizer, build_criterion, build_optimizer_with_groups
from stage2 import (get_base_model, find_classifier_layers, freeze_backbone_params,
                    unfreeze_all_params, reinit_classifier_layers, set_batchnorm_eval,
                    build_stage2_loader, apply_tau_norm_to_classifier)
from train_eval import train_one_epoch, evaluate_with_analysis
from trainer_logging import TrainingLogger
from analysis import ClassificationAnalyzer
from training_utils import create_early_stopping, ModelCheckpointer, TrainingManager, MetricsTracker
from common import setup_seed, parse_gpu_ids, setup_device
from visualization import visualize_all_results


# =============================================================================
# 时间格式化辅助函数
# =============================================================================

def format_time_ms(seconds: float) -> str:
    """将秒转换为毫秒并格式化为 X.XX ms"""
    return f"{seconds * 1000:.2f} ms"


# =============================================================================
# 简化的实验目录管理
# =============================================================================

def setup_experiment_dirs(exp_name: str, base_dir: str = "experiments"):
    """创建简单的实验目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")

    logs_dir = os.path.join(exp_dir, "logs")
    results_dir = os.path.join(exp_dir, "results")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ckpt2_dir = os.path.join(exp_dir, "checkpoints_stage2")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ckpt2_dir, exist_ok=True)

    # 创建latest软链接
    latest_link = os.path.join(base_dir, f"{exp_name}_latest")
    try:
        if os.path.exists(latest_link) or os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(os.path.basename(exp_dir), latest_link)
    except:
        pass

    print(f"\n{'=' * 80}")
    print(f"实验目录: {exp_dir}")
    print(f"快速访问: {latest_link}")
    print(f"{'=' * 80}\n")

    return exp_dir, logs_dir, results_dir, ckpt_dir, ckpt2_dir


def save_config(cfg: DictConfig, exp_dir: str):
    """保存配置文件"""
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    json_path = os.path.join(exp_dir, "config.json")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"配置已保存: {json_path}")


def save_final_results(results: Dict, results_dir: str):
    """保存最终结果"""
    results_path = os.path.join(results_dir, "results.json")

    # 确保可序列化
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return obj

    serializable_results = make_serializable(results)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存: {results_path}")


def save_summary_table(test_analysis: Dict, results_dir: str, timing_info: Dict = None):
    """
    保存概览表格（包含时间信息）

    Args:
        test_analysis: 测试分析结果
        results_dir: 结果保存目录
        timing_info: 时间信息字典（可选）
    """
    summary_path = os.path.join(results_dir, "summary.txt")

    overall = test_analysis['overall']
    groups = test_analysis['group_wise']
    per_class = test_analysis['per_class']

    # 计算Mean Per-Class Accuracy (mAcc)
    class_recalls = [per_class[str(i)]['recall'] for i in range(len(per_class))]
    mAcc = np.mean(class_recalls)

    # 计算Head vs Tail调和平均
    head_classes = groups.get('majority', {})
    tail_classes = groups.get('minority', {})

    head_acc = head_classes.get('accuracy', 0.0) if head_classes else 0.0
    tail_acc = tail_classes.get('accuracy', 0.0) if tail_classes else 0.0

    if head_acc + tail_acc > 0:
        harmonic_mean = 2 * head_acc * tail_acc / (head_acc + tail_acc)
    else:
        harmonic_mean = 0.0

    # 生成表格
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("实验结果概览\n")
        f.write("=" * 60 + "\n\n")

        f.write("【整体指标】\n")
        f.write(f"  Overall Accuracy (OA)        : {overall['accuracy']:6.2f}%\n")
        f.write(f"  Mean Per-Class Accuracy (mAcc): {mAcc:6.2f}%\n")
        f.write(f"  Macro-F1                     : {overall['macro_f1']:6.2f}%\n")
        f.write(f"  Balanced Accuracy            : {overall['balanced_accuracy']:6.2f}%\n")
        f.write("\n")

        f.write("【分组指标】\n")
        for group_name in ['majority', 'medium', 'minority']:
            group_data = groups.get(group_name, {})
            if group_data:
                f.write(f"  {group_name.capitalize():8s}  Acc: {group_data['accuracy']:6.2f}%  "
                        f"F1: {group_data['f1']:6.2f}%  "
                        f"Support: {group_data['support']:5d}\n")
        f.write("\n")

        f.write("【Head vs Tail】\n")
        if head_classes:
            f.write(f"  Head (Many)      : {head_acc:6.2f}%\n")
        if tail_classes:
            f.write(f"  Tail (Few)       : {tail_acc:6.2f}%\n")
        f.write(f"  Harmonic Mean    : {harmonic_mean:6.2f}%\n")
        f.write("\n")

        # 添加时间信息
        if timing_info:
            f.write("【时间统计】\n")

            # 获取实际epochs（如果存在）
            stage1_epochs = timing_info.get('stage1_actual_epochs', '?')
            stage2_epochs = timing_info.get('stage2_actual_epochs', '?')
            total_epochs = timing_info.get('total_epochs', '?')

            # Stage-1
            f.write(f"  Stage-1 训练     : {timing_info['stage1_train_formatted']}")
            if stage1_epochs != '?':
                f.write(f" ({stage1_epochs} epochs)")
            f.write("\n")

            # Stage-2（只有当存在时才显示）
            if timing_info.get('stage2_train_ms', 0) > 0:
                f.write(f"  Stage-2 训练     : {timing_info['stage2_train_formatted']}")
                if stage2_epochs != '?':
                    f.write(f" ({stage2_epochs} epochs)")
                f.write("\n")

            # 其他时间
            f.write(f"  验证             : {timing_info['validation_formatted']}\n")
            f.write(f"  测试             : {timing_info['test_formatted']}\n")
            f.write(f"  测试吞吐量       : {timing_info['test_throughput']:.2f} samples/sec\n")
            f.write(f"  其他             : {timing_info['other_formatted']}\n")
            f.write(f"  ───────────────────────────────────\n")
            f.write(f"  总时间           : {timing_info['total_formatted']}\n")

            # 平均每epoch
            f.write(f"  平均每epoch      : {timing_info['avg_epoch_ms']:.2f} ms")
            if total_epochs != '?':
                planned_epochs = timing_info.get('planned_epochs', '?')
                if planned_epochs != '?':
                    f.write(f" (实际 {total_epochs}/{planned_epochs} epochs)")
            f.write("\n")
            f.write("\n")

        f.write("=" * 60 + "\n")

    print(f"概览表格已保存: {summary_path}")

    # 也打印到控制台
    print("\n" + "=" * 60)
    print("实验结果概览")
    print("=" * 60)
    print(f"\nOA: {overall['accuracy']:.2f}%  |  mAcc: {mAcc:.2f}%  |  Macro-F1: {overall['macro_f1']:.2f}%")
    print(f"Many: {head_acc:.2f}%  |  Few: {tail_acc:.2f}%  |  HM: {harmonic_mean:.2f}%")

    if timing_info:
        print(f"\n训练时间: {timing_info['stage1_train_formatted']}", end='')
        if timing_info.get('stage2_train_ms', 0) > 0:
            print(f" + {timing_info['stage2_train_formatted']}", end='')
        print(f"  |  总时间: {timing_info['total_formatted']}")

    print("=" * 60)


# =============================================================================
# 主训练流程
# =============================================================================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """主训练函数"""

    # 添加总计时器
    experiment_start_time = time.time()

    # 时间记录字典
    timing_info = {
        'stage1_train': 0.0,
        'stage2_train': 0.0,
        'validation': 0.0,
        'test': 0.0,
        'visualization': 0.0,
        'total': 0.0
    }

    # 1. 初始化
    setup_seed(cfg.seed)
    gpu_ids = parse_gpu_ids(cfg.gpus)
    device = setup_device(cfg.device, gpu_ids=gpu_ids)

    # 2. 创建实验目录
    exp_dir, logs_dir, results_dir, ckpt_dir, ckpt2_dir = setup_experiment_dirs(cfg.exp_name)
    save_config(cfg, exp_dir)

    # 3. 设置日志
    log_file = os.path.join(logs_dir, "training.log")
    console_log = os.path.join(logs_dir, "console.log")

    # 重定向控制台输出到文件
    class TeeLogger:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    console_file = open(console_log, 'w', encoding='utf-8')
    sys.stdout = TeeLogger(sys.stdout, console_file)
    sys.stderr = TeeLogger(sys.stderr, console_file)

    logger = TrainingLogger(log_file, cfg.get('console_log_interval', 1))

    print("\n配置概览:")
    print(f"  模型: {cfg.model.name}")
    print(f"  损失: {cfg.loss.name}")
    print(f"  采样: {cfg.sampling.name}")
    print(f"  学习率: {cfg.training.lr}")
    print(f"  批大小: {cfg.data.batch_size}")
    print(f"  两阶段训练: {'是' if cfg.stage2.enabled else '否'}")

    # 4. 数据准备
    print("\n=== 数据准备 ===")
    loader_cfg = LoaderConfig(
        path_train=cfg.data.path_train,
        path_val=cfg.data.path_val,
        path_test=cfg.data.path_test,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        target_length=cfg.data.target_length,
        sampler=cfg.sampling.name if cfg.sampling.name != 'none' else None,
        seed=cfg.seed,
    )

    loaders = build_dataloaders(loader_cfg)
    train_dataset = loaders['train_dataset']
    num_classes = loaders['num_classes']

    # 处理验证集
    if cfg.data.path_val is None and loaders['val'] is None:
        print(f"未提供验证集，从训练集划分 {cfg.data.val_ratio * 100:.0f}%...")
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset

        labels = train_dataset.labels
        indices = np.arange(len(train_dataset))

        train_indices, val_indices = train_test_split(
            indices, test_size=cfg.data.val_ratio,
            stratify=labels, random_state=cfg.seed
        )

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        train_subset.labels = labels[train_indices]
        val_subset.labels = labels[val_indices]

        train_loader = DataLoader(
            train_subset, batch_size=cfg.data.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=cfg.data.batch_size,
            shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True
        )

        class_counts = np.bincount(train_subset.labels, minlength=num_classes)
    else:
        train_loader = loaders['train']
        val_loader = loaders['val']
        class_counts = loaders['class_counts']

    test_loader = loaders['test']

    # 创建人工不平衡（如需要）
    if cfg.create_imbalance:
        print(f"\n创建人工不平衡 (ratio: {cfg.data.imbalance_ratio})")
        longtail_indices = make_long_tailed_indices(
            train_dataset.labels, num_classes,
            cfg.data.imbalance_ratio, cfg.seed
        )

        imbalanced_dataset = ADSBSignalDataset(
            path=cfg.data.path_train,
            indices=longtail_indices,
            target_length=cfg.data.target_length,
            seed=cfg.seed
        )

        train_loader = DataLoader(
            imbalanced_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        class_counts = imbalanced_dataset.class_counts

    print(f"\n数据统计:")
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  测试样本: {len(test_loader.dataset)}")
    print(f"  类别数: {num_classes}")
    print(f"  类别分布: {class_counts.tolist()}")
    print(f"  不平衡比: {class_counts.max() / max(1, class_counts.min()):.2f}")

    # 5. 创建分析器
    analyzer = ClassificationAnalyzer(
        class_counts=class_counts,
        grouping=cfg.experiment.grouping,
        many_thresh=cfg.experiment.many_thresh,
        few_thresh=cfg.experiment.few_thresh,
    )

    # 6. 模型和优化器
    print("\n=== 模型设置 ===")
    model_kwargs = {
        'num_classes': num_classes,
        'dropout_rate': cfg.model.dropout,
    }
    if hasattr(cfg.model, 'use_attention'):
        model_kwargs['use_attention'] = cfg.model.use_attention
    if hasattr(cfg.model, 'norm_kind'):
        model_kwargs['norm_kind'] = cfg.model.norm_kind

    base_model = create_model(cfg.model.name, **model_kwargs)

    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(base_model.to(device), device_ids=gpu_ids)
    else:
        model = base_model.to(device)

    criterion = build_criterion(cfg.loss.name, cfg, class_counts).to(device)
    optimizer = build_optimizer(cfg.training.optimizer, model.parameters(),
                                cfg.training.lr, cfg.training.weight_decay)

    stage1_epochs = cfg.training.epochs
    scheduler = build_scheduler_for_stage(optimizer, cfg, stage1_epochs, 'stage1')

    # 7. 早停和检查点
    early_stopping = create_early_stopping(
        patience=cfg.early_stopping.patience,
        monitor='loss' if 'loss' in cfg.early_stopping.monitor else 'acc',
        mode=cfg.early_stopping.mode,
        save_path=os.path.join(ckpt_dir, 'best.pth'),
        verbose=True
    )

    checkpointer = ModelCheckpointer(
        save_dir=ckpt_dir,
        save_best=True,
        monitor='val_loss' if 'loss' in cfg.early_stopping.monitor else 'val_acc',
        mode=cfg.early_stopping.mode
    )

    manager = TrainingManager(
        model=model, optimizer=optimizer, scheduler=scheduler,
        early_stopping=early_stopping, checkpointer=checkpointer,
        metrics_tracker=MetricsTracker()
    )

    # 8. Stage-1 训练
    print(f"\n=== Stage-1 训练 ({stage1_epochs} epochs) ===")
    best_val_acc_stage1 = 0.0
    scaler = GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

    # 添加实际epoch跟踪
    actual_stage1_epochs = 0

    # 添加 Stage-1计时
    stage1_start_time = time.time()
    val_time_stage1 = 0.0

    for epoch in range(1, stage1_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logger.start_epoch(epoch, lr=current_lr)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            logger, epoch, cfg.training.grad_clip, cfg.amp, scaler
        )

        # 验证（记录时间）
        val_start = time.time()
        val_metrics, val_analysis, _, _, val_timing = evaluate_with_analysis(
            model, val_loader, criterion, device, analyzer, class_counts,
            cfg.evaluation.eval_logit_adjust, cfg.evaluation.eval_logit_tau
        )
        val_time_stage1 += (time.time() - val_start)

        metrics = {
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'val_balanced_acc': val_metrics['balanced_acc']
        }
        manager.step_epoch(metrics)

        logger.log_epoch_end(epoch, train_metrics, val_metrics,
                             val_analysis['group_wise'], optimizer.param_groups[0]['lr'])

        # 记录实际运行的epoch
        actual_stage1_epochs = epoch

        if manager.should_stop_training():
            print(f"\n[Stage-1] 早停于 epoch {epoch}")
            break

        best_val_acc_stage1 = max(best_val_acc_stage1, val_metrics['acc'])

    # 计算Stage-1总时间
    stage1_total_time = time.time() - stage1_start_time
    stage1_train_time = stage1_total_time - val_time_stage1

    timing_info['stage1_train'] = stage1_train_time
    timing_info['validation'] += val_time_stage1

    # 打印Stage-1时间统计
    print(f"\n[Stage-1 时间统计]")
    print(f"  训练时间: {format_time_ms(stage1_train_time)}")
    print(f"  验证时间: {format_time_ms(val_time_stage1)}")
    print(f"  总时间: {format_time_ms(stage1_total_time)}")
    print(f"  平均每epoch: {(stage1_total_time / max(1, actual_stage1_epochs)) * 1000:.2f} ms")

    # 加载最佳Stage-1模型
    best_path = os.path.join(ckpt_dir, 'best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[Stage-1] 已加载最佳模型")

    # 9. Stage-2 训练（可选）
    best_val_acc_stage2 = None
    criterion2 = criterion  # 默认值，防止后面引用错误
    stage2_epochs = 0
    actual_stage2_epochs = 0

    if cfg.stage2.enabled and cfg.stage2.mode != 'tau_norm_only':
        stage2_epochs = cfg.stage2.epochs or int(0.3 * stage1_epochs)
        print(f"\n=== Stage-2 训练 ({stage2_epochs} epochs) ===")
        print(f"模式: {cfg.stage2.mode}")

        base = get_base_model(model)
        clf_pairs = find_classifier_layers(base, num_classes)
        classifier_names = [n for n, _ in clf_pairs]
        classifier_layers = [m for _, m in clf_pairs]

        # 定义stage2参数
        stage2_wd = cfg.stage2.weight_decay if hasattr(cfg.stage2,
                                                       'weight_decay') and cfg.stage2.weight_decay is not None else cfg.training.weight_decay

        # 创建Stage-2的loss
        criterion2 = build_criterion(
            cfg.stage2.loss or cfg.loss.name, cfg, class_counts
        ).to(device)

        # 构建Stage-2 loader
        # 构建Stage-2采样配置
        if cfg.stage2.sampler is None:
            # 继承Stage-1的采样配置
            stage2_sampler_config = {
                'name': cfg.sampling.name,
                'alpha': getattr(cfg.sampling, 'alpha', 0.5),
                'alpha_start': getattr(cfg.sampling, 'alpha_start', 0.5),
                'alpha_end': getattr(cfg.sampling, 'alpha_end', 0.0),
            }
            print("[Stage-2] 采样策略: 继承Stage-1配置")
        else:
            # 使用Stage-2指定的采样配置
            stage2_sampler_config = {
                'name': cfg.stage2.sampler,
                'alpha': getattr(cfg.stage2, 'alpha', 0.5),
                'alpha_start': getattr(cfg.stage2, 'alpha_start', 0.5),
                'alpha_end': getattr(cfg.stage2, 'alpha_end', 0.0),
            }
            print(f"[Stage-2] 采样策略: {cfg.stage2.sampler}")

        # 构建Stage-2 loader
        stage2_loader = build_stage2_loader(
            train_loader.dataset,
            cfg.data.batch_size,
            cfg.data.num_workers,
            stage2_sampler_config,
            total_epochs=stage2_epochs,
            seed=cfg.seed
        )

        # 获取sampler引用（用于epoch更新）
        stage2_sampler = getattr(stage2_loader, 'sampler', None)

        # BN处理（在创建optimizer之前）
        if cfg.stage2.freeze_bn:
            if cfg.stage2.mode == 'crt':
                print("[Stage-2] 冻结BN层统计量（CRT模式）")
                base.apply(set_batchnorm_eval)
            elif cfg.stage2.sampler == 'same':
                print("[Stage-2] 冻结BN层统计量（采样分布不变）")
                base.apply(set_batchnorm_eval)
            else:
                print("[Stage-2] BN层保持训练模式（finetune + 平衡采样）")

        # 添加 Stage-2计时
        stage2_start_time = time.time()
        val_time_stage2 = 0.0

        # 冻结/重初始化
        if cfg.stage2.mode == 'crt':
            print("[Stage-2] CRT: 冻结backbone，重初始化分类器")
            freeze_backbone_params(base, classifier_names)
            reinit_classifier_layers(classifier_layers)
            params_to_optimize = [p for p in base.parameters() if p.requires_grad]

            stage2_lr = cfg.stage2.lr or cfg.training.lr
            print(f"[Stage-2] 学习率: {stage2_lr}")

            optimizer2 = build_optimizer(
                cfg.stage2.optimizer or cfg.training.optimizer,
                params_to_optimize, stage2_lr, stage2_wd
            )

        else:  # finetune
            print("[Stage-2] Finetune: 解冻所有参数，使用差异化学习率")
            unfreeze_all_params(base)

            # 构建差异化学习率参数组
            backbone_params = []
            classifier_params = []

            for name, param in base.named_parameters():
                if any(name.startswith(cn) for cn in classifier_names):
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

            stage2_lr = cfg.stage2.lr if hasattr(cfg.stage2, 'lr') and cfg.stage2.lr is not None else (
                    cfg.training.lr * 0.1)
            clf_lr_mult = cfg.stage2.clf_lr_mult if hasattr(cfg.stage2, 'clf_lr_mult') else 5.0

            print(f"[Stage-2] Backbone学习率: {stage2_lr}")
            print(f"[Stage-2] 分类器学习率: {stage2_lr * clf_lr_mult} (倍数: {clf_lr_mult})")

            param_groups = [
                {'params': backbone_params, 'lr': stage2_lr},
                {'params': classifier_params, 'lr': stage2_lr * clf_lr_mult}
            ]

            optimizer2 = build_optimizer_with_groups(
                cfg.stage2.optimizer or cfg.training.optimizer,
                param_groups, stage2_wd
            )

        scheduler2 = build_scheduler_for_stage(optimizer2, cfg, stage2_epochs, 'stage2')

        early_stopping2 = create_early_stopping(
            patience=cfg.early_stopping.patience,
            monitor='loss' if 'loss' in cfg.early_stopping.monitor else 'acc',
            mode=cfg.early_stopping.mode,
            save_path=os.path.join(ckpt2_dir, 'best.pth'),
            verbose=True
        )

        checkpointer2 = ModelCheckpointer(
            save_dir=ckpt2_dir,
            save_best=True,
            monitor='val_loss' if 'loss' in cfg.early_stopping.monitor else 'val_acc',
            mode=cfg.early_stopping.mode
        )

        manager2 = TrainingManager(
            model=model, optimizer=optimizer2, scheduler=scheduler2,
            early_stopping=early_stopping2, checkpointer=checkpointer2,
            metrics_tracker=MetricsTracker()
        )

        scaler2 = GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

        for epoch in range(1, stage2_epochs + 1):
            if stage2_sampler is not None and hasattr(stage2_sampler, 'set_epoch'):
                stage2_sampler.set_epoch(epoch - 1)
                if hasattr(stage2_sampler, 'get_progress_info'):
                    info = stage2_sampler.get_progress_info()
                    print(
                        f"[Stage-2 Progressive] Epoch {epoch}: alpha={info['current_alpha']:.3f}, progress={info['progress']:.1%}")

            current_lr = optimizer2.param_groups[0]['lr']
            logger.start_epoch(stage1_epochs + epoch, lr=current_lr)

            train_metrics = train_one_epoch(
                model, stage2_loader, criterion2, optimizer2, device,
                logger, stage1_epochs + epoch, cfg.training.grad_clip, cfg.amp, scaler2
            )

            # 验证（记录时间）
            val_start = time.time()
            val_metrics, val_analysis, _, _, val_timing = evaluate_with_analysis(
                model, val_loader, criterion2, device, analyzer, class_counts,
                cfg.evaluation.eval_logit_adjust, cfg.evaluation.eval_logit_tau
            )
            val_time_stage2 += (time.time() - val_start)

            metrics = {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
                'val_balanced_acc': val_metrics['balanced_acc']
            }
            manager2.step_epoch(metrics)

            logger.log_epoch_end(stage1_epochs + epoch, train_metrics, val_metrics,
                                 val_analysis['group_wise'], optimizer2.param_groups[0]['lr'])

            # 记录实际运行的epoch
            actual_stage2_epochs = epoch

            if manager2.should_stop_training():
                print(f"\n[Stage-2] 早停于 epoch {epoch}")
                break

            best_val_acc_stage2 = val_metrics['acc'] if best_val_acc_stage2 is None else \
                max(best_val_acc_stage2, val_metrics['acc'])

        # 计算Stage-2总时间
        stage2_total_time = time.time() - stage2_start_time
        stage2_train_time = stage2_total_time - val_time_stage2

        timing_info['stage2_train'] = stage2_train_time
        timing_info['validation'] += val_time_stage2

        # 打印Stage-2时间统计
        print(f"\n[Stage-2 时间统计]")
        print(f"  训练时间: {format_time_ms(stage2_train_time)}")
        print(f"  验证时间: {format_time_ms(val_time_stage2)}")
        print(f"  总时间: {format_time_ms(stage2_total_time)}")
        print(f"  平均每epoch: {(stage2_total_time / max(1, actual_stage2_epochs)) * 1000:.2f} ms")

        # 加载最佳Stage-2模型
        best_path2 = os.path.join(ckpt2_dir, 'best.pth')
        if os.path.exists(best_path2):
            checkpoint = torch.load(best_path2, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[Stage-2] 已加载最佳模型")

    # 10. Stage-3 校准（可选）
    if cfg.stage3.mode in ('tau_norm', 'both'):
        base = get_base_model(model)
        clf_pairs = find_classifier_layers(base, num_classes)
        classifier_layers = [m for _, m in clf_pairs]
        apply_tau_norm_to_classifier(classifier_layers, cfg.stage3.tau_norm)
        print(f"[Stage-3] τ-normalization应用 (tau={cfg.stage3.tau_norm})")

    # 11. 最终测试
    print("\n=== 最终测试 ===")

    # 测试计时
    test_start_time = time.time()

    # 选择正确的criterion
    if cfg.stage2.enabled and cfg.stage2.mode != 'tau_norm_only':
        test_criterion = criterion2
    else:
        test_criterion = criterion

    eval_mode = cfg.evaluation.eval_logit_adjust
    eval_tau = cfg.evaluation.eval_logit_tau
    if cfg.stage3.mode in ('logit_adjust', 'both'):
        eval_mode = 'posthoc'
        eval_tau = cfg.stage3.logit_tau

    test_metrics, test_analysis, _, _, test_timing = evaluate_with_analysis(
        model, test_loader, test_criterion, device, analyzer,
        class_counts, eval_mode, eval_tau
    )

    test_total_time = time.time() - test_start_time
    timing_info['test'] = test_total_time

    # 12. 打印测试结果
    print(f"\n{'=' * 60}")
    print("最终测试结果")
    print(f"{'=' * 60}")
    print(f"准确率: {test_analysis['overall']['accuracy']:.2f}%")
    print(f"平衡准确率: {test_analysis['overall']['balanced_accuracy']:.2f}%")
    print(f"Macro F1: {test_analysis['overall']['macro_f1']:.2f}%")
    print(f"\n测试时间: {format_time_ms(test_total_time)}")
    print(f"吞吐量: {test_timing['throughput_samples_per_sec']:.2f} samples/sec")

    print("\n类别组性能:")
    for group_name, metrics in test_analysis['group_wise'].items():
        if metrics:
            print(f"  {group_name.capitalize():8s}: "
                  f"Acc={metrics['accuracy']:5.2f}% | "
                  f"F1={metrics['f1']:5.2f}% | "
                  f"Support={metrics['support']:4d}")

    # 13. 保存结果（添加时间信息）
    # 计算总时间
    experiment_end_time = time.time()
    timing_info['total'] = experiment_end_time - experiment_start_time

    # 其他时间（数据加载、初始化等）
    timing_info['other'] = (timing_info['total'] -
                            timing_info['stage1_train'] -
                            timing_info['stage2_train'] -
                            timing_info['validation'] -
                            timing_info['test'])

    # 计算实际总epochs
    actual_total_epochs = actual_stage1_epochs + actual_stage2_epochs

    # 格式化时间信息
    timing_formatted = {
        'stage1_train_ms': float(timing_info['stage1_train'] * 1000),
        'stage1_train_formatted': format_time_ms(timing_info['stage1_train']),
        'stage1_actual_epochs': actual_stage1_epochs,

        'stage2_train_ms': float(timing_info['stage2_train'] * 1000),
        'stage2_train_formatted': format_time_ms(timing_info['stage2_train']),
        'stage2_actual_epochs': actual_stage2_epochs,

        'validation_ms': float(timing_info['validation'] * 1000),
        'validation_formatted': format_time_ms(timing_info['validation']),

        'test_ms': float(timing_info['test'] * 1000),
        'test_formatted': format_time_ms(timing_info['test']),
        'test_throughput': float(test_timing['throughput_samples_per_sec']),

        'other_ms': float(timing_info['other'] * 1000),
        'other_formatted': format_time_ms(timing_info['other']),

        'total_ms': float(timing_info['total'] * 1000),
        'total_formatted': format_time_ms(timing_info['total']),

        'total_epochs': actual_total_epochs,
        'planned_epochs': stage1_epochs + stage2_epochs,
        'avg_epoch_ms': (timing_info['total'] / max(1, actual_total_epochs)) * 1000
    }

    final_results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'dataset': {
            'num_classes': int(num_classes),
            'class_counts': class_counts.tolist(),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
        },
        'training': {
            'best_val_acc_stage1': float(best_val_acc_stage1),
            'best_val_acc_stage2': float(best_val_acc_stage2) if best_val_acc_stage2 else None,
            'stage1_epochs': stage1_epochs,
            'stage2_epochs': stage2_epochs,
            'total_epochs': stage1_epochs + stage2_epochs,
            'actual_stage1_epochs': actual_stage1_epochs,
            'actual_stage2_epochs': actual_stage2_epochs,
            'actual_total_epochs': actual_total_epochs,
        },
        'timing': timing_formatted,
        'test_results': test_analysis,
    }

    save_final_results(final_results, results_dir)
    save_summary_table(test_analysis, results_dir, timing_formatted)

    # 14. 最终时间统计
    print(f"\n{'=' * 80}")
    print("实验时间统计")
    print(f"{'=' * 80}")
    print(f"Stage-1 训练: {timing_formatted['stage1_train_formatted']} "
          f"({actual_stage1_epochs} epochs)")
    if cfg.stage2.enabled:
        print(f"Stage-2 训练: {timing_formatted['stage2_train_formatted']} "
              f"({actual_stage2_epochs} epochs)")
    print(f"验证总计: {timing_formatted['validation_formatted']}")
    print(f"测试: {timing_formatted['test_formatted']} "
          f"({timing_formatted['test_throughput']:.2f} samples/sec)")
    if cfg.visualization.enabled and timing_info['visualization'] > 0:
        print(f"可视化: {format_time_ms(timing_info['visualization'])}")
    print(f"其他: {timing_formatted['other_formatted']}")
    print(f"{'─' * 80}")
    print(f"总时间: {timing_formatted['total_formatted']}")
    print(f"平均每epoch: {timing_formatted['avg_epoch_ms']:.2f} ms "
          f"(实际 {actual_total_epochs}/{stage1_epochs + stage2_epochs} epochs)")
    print(f"{'=' * 80}\n")

    print(f"\n实验完成！")
    print(f"实验目录: {exp_dir}")
    print(f"日志: {logs_dir}")
    print(f"结果: {results_dir}")
    print(f"检查点: {ckpt_dir}")

    return exp_dir


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)