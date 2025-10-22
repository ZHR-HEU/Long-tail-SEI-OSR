# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from common import tau_norm_weights

def get_base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def find_classifier_layers(model: nn.Module, num_classes: int) -> List[Tuple[str, nn.Module]]:
    candidates: List[Tuple[str, nn.Module]] = []
    last_linear: Optional[Tuple[str, nn.Linear]] = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = (name, m)
            if m.out_features == num_classes:
                candidates.append((name, m))
    return candidates if candidates else ([last_linear] if last_linear is not None else [])

def freeze_backbone_params(model: nn.Module, classifier_names: Iterable[str]):
    cls = list(classifier_names)
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(cn) for cn in cls)

def unfreeze_all_params(model: nn.Module):
    for p in model.parameters(): p.requires_grad = True

def reinit_classifier_layers(layers: List[nn.Module]):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def apply_tau_norm_to_classifier(layers: List[nn.Module], tau: float = 1.0):
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(tau_norm_weights(layer.weight, tau=tau))

def set_batchnorm_eval(module: nn.Module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        module.eval()


def build_stage2_loader(
        dataset,
        batch_size: int,
        num_workers: int,
        sampler_config: Dict,  # 统一的采样配置
        total_epochs: int = 100,
        seed: int = 0
):
    """
    构建Stage-2数据加载器，支持所有采样策略

    Args:
        dataset: 数据集
        batch_size: 批大小
        num_workers: 工作进程数
        sampler_config: 采样配置字典，包含：
            - name: 采样器名称
            - alpha: power采样的alpha参数
            - alpha_start/alpha_end: progressive采样参数
        total_epochs: 总epoch数（用于progressive采样）
        seed: 随机种子
    """
    from data_utils import make_sampler

    if not hasattr(dataset, 'labels'):
        raise AttributeError("Stage-2 sampler requires dataset.labels")

    labels = np.asarray(dataset.labels).astype(int)
    sampler_name = sampler_config.get('name', 'none')

    # 特殊处理："same"表示使用原始分布
    if sampler_name == 'same':
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # 使用统一的采样器工厂函数
    sampler = make_sampler(
        labels=labels,
        method=sampler_name,
        seed=seed,
        alpha=sampler_config.get('alpha', 0.5),
        alpha_start=sampler_config.get('alpha_start', 0.5),
        alpha_end=sampler_config.get('alpha_end', 0.0),
        total_epochs=total_epochs
    )

    # 如果返回None（表示不使用采样器），则使用shuffle
    if sampler is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )