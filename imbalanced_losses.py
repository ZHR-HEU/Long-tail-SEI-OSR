# -*- coding: utf-8 -*-
"""
Optimized loss functions for imbalanced learning with enhanced features.

Design Philosophy
----------------
1. CLI-friendly: All losses support both dict-based and keyword argument initialization
2. Unified interface: Consistent forward signature across all losses
3. Memory efficient: Lazy weight computation and caching
4. Numerically stable: Safe operations with proper epsilon handling
5. Extensible: Easy to add new losses and combine existing ones

Features
--------
- Enhanced base classes with better error handling
- Improved numerical stability and device handling
- Loss combination and scheduling capabilities
- Advanced weighting strategies
- Cost-sensitive objectives (expected cost, cost-weighted CE, cost-sensitive focal)
- Comprehensive validation and logging

Notes
-----
- Logit-Adjustment 的思路可在“模型头（减 tau*log π）”或“损失函数（加 tau*log π）”
  两处实现；请**二选一**避免重复调整。
- 所有基于类计数/代价矩阵的损失都会在前向时校验 num_classes 一致性。
- 期望代价（Expected Cost）等价于最小化贝叶斯风险：loss_i = sum_j C[y_i, j] * p(j|x_i)。

Dependencies: torch, numpy

Author: Enhanced Implementation (revised + cost-sensitive)
Date: 2025
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Union, Callable, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Core Utilities and Helpers
# =============================================================================

@dataclass
class LossConfig:
    """Configuration container for loss functions."""
    reduction: str = 'mean'
    eps: float = 1e-8
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"Invalid label_smoothing: {self.label_smoothing}")


class TensorUtils:
    """Utility functions for tensor operations."""

    @staticmethod
    def to_1d_long(x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to 1D long tensor."""
        return x.long().view(-1)

    @staticmethod
    def as_tensor1d(
        x: Union[Sequence[int], np.ndarray, torch.Tensor],
        device=None,
        dtype=None
    ) -> torch.Tensor:
        """Convert input to 1D tensor with optional device/dtype."""
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)

        if t.dim() != 1:
            t = t.view(-1)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t

    @staticmethod
    def as_tensor2d(
        x: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        device=None,
        dtype=None
    ) -> torch.Tensor:
        """Convert input to 2D tensor with optional device/dtype."""
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if t.dim() != 2:
            raise ValueError(f"Expect 2D tensor for cost matrix, got shape {tuple(t.shape)}")
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t

    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Numerically stable logarithm."""
        return torch.log(x.clamp_min(eps))

    @staticmethod
    def safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Numerically stable division."""
        return numerator / (denominator + eps)

    @staticmethod
    def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert labels to one-hot encoding."""
        labels = TensorUtils.to_1d_long(labels)
        y = torch.zeros(labels.size(0), num_classes, device=labels.device, dtype=torch.float)
        y.scatter_(1, labels.view(-1, 1), 1.0)
        return y


class WeightComputer:
    """Compute various types of class weights for imbalanced learning."""

    @staticmethod
    def class_priors(
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        device=None,
        eps: float = 1e-12
    ) -> torch.Tensor:
        """Compute class priors from counts."""
        counts = TensorUtils.as_tensor1d(class_counts, device=device, dtype=torch.float)
        total = counts.sum()
        if total <= 0:
            raise ValueError("class_priors: sum(class_counts) must be > 0")
        priors = counts / total
        return priors.clamp_min(eps)

    @staticmethod
    def inverse_frequency_weights(
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        gamma: float = 1.0,
        normalize: bool = True,
        device=None
    ) -> torch.Tensor:
        """Compute inverse frequency weights: 1 / n_c^gamma."""
        counts = TensorUtils.as_tensor1d(class_counts, device=device, dtype=torch.float)
        weights = 1.0 / torch.pow(counts.clamp_min(1.0), gamma)
        if normalize:
            weights = weights / weights.mean().clamp_min(1e-12)
        return weights

    @staticmethod
    def effective_number_weights(
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        beta: float = 0.9999,
        normalize: bool = True,
        device=None
    ) -> torch.Tensor:
        """Compute effective number weights: (1-β) / (1-β^n_c)."""
        counts = TensorUtils.as_tensor1d(class_counts, device=device, dtype=torch.float)
        counts = counts.clamp_min(1.0)  # 防止 n_c=0
        beta_tensor = torch.tensor(beta, device=counts.device, dtype=counts.dtype)

        effective_num = 1.0 - torch.pow(beta_tensor, counts)
        weights = (1.0 - beta_tensor) / effective_num.clamp_min(1e-12)  # 分母保护

        if normalize:
            weights = weights / weights.mean().clamp_min(1e-12)
        return weights

    @staticmethod
    def progressive_weights(
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        epoch: int,
        total_epochs: int,
        start_weights: str = 'uniform',
        end_weights: str = 'inverse',
        device=None
    ) -> torch.Tensor:
        """Compute progressive weights that change over training."""
        progress = min(epoch / max(total_epochs, 1), 1.0)

        # Get start weights
        if start_weights == 'uniform':
            w_start = torch.ones(len(class_counts), device=device)
        elif start_weights == 'inverse':
            w_start = WeightComputer.inverse_frequency_weights(class_counts, device=device)
        elif start_weights == 'effective':
            w_start = WeightComputer.effective_number_weights(class_counts, device=device)
        else:
            raise ValueError(f"Unknown start_weights: {start_weights}")

        # Get end weights
        if end_weights == 'uniform':
            w_end = torch.ones(len(class_counts), device=device)
        elif end_weights == 'inverse':
            w_end = WeightComputer.inverse_frequency_weights(class_counts, device=device)
        elif end_weights == 'effective':
            w_end = WeightComputer.effective_number_weights(class_counts, device=device)
        else:
            raise ValueError(f"Unknown end_weights: {end_weights}")

        # Linear interpolation
        weights = (1 - progress) * w_start + progress * w_end
        return weights / weights.mean().clamp_min(1e-12)


# =============================================================================
# Enhanced Base Classes
# =============================================================================

class BaseLoss(nn.Module, ABC):
    """Enhanced base class for all loss functions."""

    def __init__(
        self,
        reduction: str = 'mean',
        eps: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate loss parameters."""
        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def update_epoch(self, epoch: int):
        """Update loss for new epoch (for progressive losses)."""
        pass

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass


class WeightedBaseLoss(BaseLoss):
    """Base class for losses that use class weights."""

    def __init__(
        self,
        class_counts: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
        weight_strategy: str = 'none',
        weight_params: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_counts = class_counts
        self.weight_strategy = weight_strategy
        self.weight_params = weight_params or {}
        self._cached_weights: Optional[torch.Tensor] = None
        self._cached_device = None

    def _get_weights(self, device) -> Optional[torch.Tensor]:
        """Get class weights, computing and caching if necessary."""
        if self.weight_strategy == 'none':
            return None

        if self.class_counts is None:
            warnings.warn("Class counts not provided, cannot compute weights")
            return None

        # Check if we need to recompute (device change)
        if self._cached_weights is None or self._cached_device != device:
            self._cached_weights = self._compute_weights(device)
            self._cached_device = device

        return self._cached_weights

    def _compute_weights(self, device) -> torch.Tensor:
        """Compute weights based on strategy."""
        if self.weight_strategy == 'inverse':
            return WeightComputer.inverse_frequency_weights(
                self.class_counts, device=device, **self.weight_params)
        elif self.weight_strategy == 'effective':
            return WeightComputer.effective_number_weights(
                self.class_counts, device=device, **self.weight_params)
        else:
            raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")


# =============================================================================
# Core Loss Functions
# =============================================================================

class CrossEntropy(BaseLoss):
    """Enhanced cross-entropy loss with optional label smoothing."""

    def __init__(
        self,
        weight: Optional[Union[Sequence[float], torch.Tensor]] = None,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self._weight = None
        if weight is not None:
            self._weight = TensorUtils.as_tensor1d(weight, dtype=torch.float)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)

        # Move weight to device if needed
        weight = None
        if self._weight is not None:
            weight = self._weight.to(logits.device)

        return F.cross_entropy(
            logits, targets,
            weight=weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(BaseLoss):
    """Enhanced Focal Loss with better numerical stability."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self._alpha = None
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                self._alpha = torch.tensor(alpha, dtype=torch.float)
            else:
                self._alpha = TensorUtils.as_tensor1d(alpha, dtype=torch.float)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)

        # Standard CE per-sample
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # pt = exp(-CE) for numerical stability
        pt = torch.exp(-ce_loss)

        # Focal weight
        focal_weight = torch.pow(1 - pt, self.gamma)

        # Alpha (scalar or per-class)
        if self._alpha is not None:
            alpha = self._alpha.to(logits.device)
            if alpha.dim() == 0:  # scalar
                alpha_t = alpha
            else:
                if alpha.numel() != logits.size(-1):
                    raise ValueError(
                        f"FocalLoss: len(alpha)={alpha.numel()} != num_classes={logits.size(-1)}"
                    )
                alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return self._apply_reduction(loss)


class ClassBalancedLoss(WeightedBaseLoss):
    """Class-Balanced Loss using effective number of samples."""

    def __init__(
        self,
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        beta: float = 0.9999,
        **kwargs
    ):
        super().__init__(
            class_counts=class_counts,
            weight_strategy='effective',
            weight_params={'beta': beta},
            **kwargs
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        weights = self._get_weights(logits.device)

        # class_counts length check
        if weights is not None and weights.numel() != logits.size(-1):
            raise ValueError(
                f"ClassBalancedLoss: len(class_counts)={weights.numel()} "
                f"!= num_classes={logits.size(-1)}"
            )

        return F.cross_entropy(
            logits, targets,
            weight=weights,
            reduction=self.reduction
        )


class LDAMLoss(WeightedBaseLoss):
    """Label-Distribution-Aware Margin Loss with DRW support."""

    def __init__(
        self,
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        max_margin: float = 0.5,
        scale: float = 30.0,
        drw_start_epoch: int = 0,
        reweight_power: float = 0.5,
        **kwargs
    ):
        super().__init__(class_counts=class_counts, **kwargs)
        self.max_margin = max_margin
        self.scale = scale
        self.drw_start_epoch = drw_start_epoch
        self.reweight_power = reweight_power
        self._margins: Optional[torch.Tensor] = None
        self._drw_weights: Optional[torch.Tensor] = None
        self.current_epoch = 0

    def _compute_margins(self, device) -> torch.Tensor:
        """Compute per-class margins."""
        if self._margins is None:
            counts = TensorUtils.as_tensor1d(self.class_counts, device=device, dtype=torch.float)
            margins = self.max_margin / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))  # 1/n_c^(1/4)
            self._margins = margins
        return self._margins.to(device)

    def _compute_drw_weights(self, device) -> torch.Tensor:
        """Compute DRW weights."""
        if self._drw_weights is None:
            weights = WeightComputer.inverse_frequency_weights(
                self.class_counts, gamma=self.reweight_power, device=device
            )
            self._drw_weights = weights
        return self._drw_weights.to(device)

    def update_epoch(self, epoch: int):
        """Update current epoch for DRW."""
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        device = logits.device

        # margin length check
        if len(self.class_counts) != logits.size(-1):
            raise ValueError(
                f"LDAMLoss: len(class_counts)={len(self.class_counts)} "
                f"!= num_classes={logits.size(-1)}"
            )

        # Apply margins to true class
        margins = self._compute_margins(device)
        margin_values = margins.gather(0, targets)

        adjusted_logits = logits.clone()
        batch_indices = torch.arange(logits.size(0), device=device)
        adjusted_logits[batch_indices, targets] -= margin_values

        # Scale
        adjusted_logits *= self.scale

        # DRW weights (after start epoch)
        weight = None
        if self.current_epoch >= self.drw_start_epoch:
            weight = self._compute_drw_weights(device)

        return F.cross_entropy(adjusted_logits, targets, weight=weight, reduction=self.reduction)


class BalancedSoftmaxLoss(WeightedBaseLoss):
    """Balanced Softmax Loss with prior-adjusted logits."""

    def __init__(
        self,
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        **kwargs
    ):
        super().__init__(class_counts=class_counts, **kwargs)
        self._log_priors: Optional[torch.Tensor] = None

    def _compute_log_priors(self, device) -> torch.Tensor:
        """Compute log priors from class counts."""
        if self._log_priors is None:
            priors = WeightComputer.class_priors(self.class_counts, device=device)
            self._log_priors = TensorUtils.safe_log(priors, self.eps)
        return self._log_priors.to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)

        log_priors = self._compute_log_priors(logits.device)
        if log_priors.numel() != logits.size(-1):
            raise ValueError(
                f"BalancedSoftmaxLoss: len(class_counts)={log_priors.numel()} "
                f"!= num_classes={logits.size(-1)}"
            )

        adjusted_logits = logits + log_priors
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)


class LogitAdjustmentLoss(BalancedSoftmaxLoss):
    """Logit Adjustment Loss with tunable prior strength."""

    def __init__(
        self,
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        tau: float = 1.0,
        **kwargs
    ):
        super().__init__(class_counts=class_counts, **kwargs)
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)

        log_priors = self._compute_log_priors(logits.device)
        if log_priors.numel() != logits.size(-1):
            raise ValueError(
                f"LogitAdjustmentLoss: len(class_counts)={log_priors.numel()} "
                f"!= num_classes={logits.size(-1)}"
            )

        # 训练期常用“加 tau*log π”；请勿与“LA 分类头(减 tau*log π)”叠加
        adjusted_logits = logits + self.tau * log_priors
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)


class ProgressiveLoss(WeightedBaseLoss):
    """Progressive loss that changes weighting strategy over epochs."""

    def __init__(
        self,
        class_counts: Union[Sequence[int], np.ndarray, torch.Tensor],
        total_epochs: int,
        start_strategy: str = 'uniform',
        end_strategy: str = 'inverse',
        **kwargs
    ):
        super().__init__(class_counts=class_counts, **kwargs)
        self.total_epochs = int(total_epochs)
        self.start_strategy = start_strategy
        self.end_strategy = end_strategy
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update current epoch and recompute weights."""
        self.current_epoch = int(epoch)
        # Force weight recomputation
        self._cached_weights = None

    def _compute_weights(self, device) -> torch.Tensor:
        """Compute progressive weights based on current epoch."""
        return WeightComputer.progressive_weights(
            self.class_counts,
            self.current_epoch,
            self.total_epochs,
            self.start_strategy,
            self.end_strategy,
            device
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        weights = self._compute_weights(logits.device)

        if weights is not None and weights.numel() != logits.size(-1):
            raise ValueError(
                f"ProgressiveLoss: len(class_counts)={weights.numel()} "
                f"!= num_classes={logits.size(-1)}"
            )

        return F.cross_entropy(logits, targets, weight=weights, reduction=self.reduction)


# =============================================================================
# Cost-Sensitive Losses
# =============================================================================

# =============================================================================
# Cost-Sensitive Losses (改进版)
# =============================================================================
# =============================================================================
# Cost-Sensitive Losses (改进版)
# =============================================================================

def _prepare_cost_vector_from_inputs(
        num_classes: int,
        device,
        *,
        cost_vector: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
        cost_matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
        cost_strategy: str = 'manual',
        class_counts: Optional[Union[Sequence[int], np.ndarray]] = None,
        zero_diagonal: bool = True,
        normalize: bool = True,
        eps: float = 1e-12
) -> torch.Tensor:
    """
    Build a per-class cost vector w_c from either vector, matrix, or auto-generation.

    Args:
        num_classes: Number of classes
        device: Target device
        cost_vector: Manual cost vector (C,)
        cost_matrix: Manual cost matrix (C, C)
        cost_strategy: 'manual', 'auto', 'uniform', 'sqrt', 'log'
        class_counts: Class sample counts for auto-generation
        zero_diagonal: Zero out diagonal for matrix
        normalize: Normalize to mean=1
        eps: Epsilon for numerical stability

    Returns:
        Cost vector (C,)
    """
    # Strategy 1: Manual specification
    if cost_strategy == 'manual':
        if cost_vector is None and cost_matrix is None:
            raise ValueError("cost_strategy='manual' requires cost_vector or cost_matrix")

        if cost_vector is not None:
            w = TensorUtils.as_tensor1d(cost_vector, device=device, dtype=torch.float)
            if w.numel() != num_classes:
                raise ValueError(f"cost_vector length {w.numel()} != num_classes {num_classes}")
        else:
            C = TensorUtils.as_tensor2d(cost_matrix, device=device, dtype=torch.float)
            if C.shape[0] != num_classes or C.shape[1] != num_classes:
                raise ValueError(f"cost_matrix must be (C,C), got {tuple(C.shape)} vs C={num_classes}")
            C = C.clone()
            if zero_diagonal:
                C.fill_diagonal_(0.0)
            # Mean off-diagonal per row
            offdiag_sum = C.sum(dim=1)
            denom = max(num_classes - 1, 1)
            w = offdiag_sum / float(denom)
            w = w.clamp_min(eps)

    # Strategy 2: Uniform (all costs equal)
    elif cost_strategy == 'uniform':
        w = torch.ones(num_classes, device=device, dtype=torch.float)

    # Strategy 3-5: Auto-generate from class_counts
    else:
        if class_counts is None:
            raise ValueError(f"cost_strategy='{cost_strategy}' requires class_counts")

        counts = np.asarray(class_counts).astype(np.float64)
        counts = np.clip(counts, 1.0, None)  # Prevent division by zero

        if cost_strategy == 'auto':
            # Inverse frequency (standard approach)
            cost_vec = 1.0 / counts

        elif cost_strategy == 'sqrt':
            # Square root inverse frequency (more gentle)
            cost_vec = 1.0 / np.sqrt(counts)

        elif cost_strategy == 'log':
            # Logarithmic (even more gentle)
            cost_vec = 1.0 / np.log1p(counts)

        else:
            raise ValueError(f"Unknown cost_strategy: {cost_strategy}. "
                             f"Valid: 'manual', 'auto', 'uniform', 'sqrt', 'log'")

        w = torch.tensor(cost_vec, device=device, dtype=torch.float)

    # Normalize to mean=1 if requested
    if normalize and cost_strategy != 'uniform':
        w = w / w.mean().clamp_min(eps)

    return w


class CostSensitiveCE(BaseLoss):
    """
    Cost-sensitive cross-entropy with automatic cost generation.

    Usage modes:
        1. Auto mode (recommended):
           loss = CostSensitiveCE(cost_strategy='auto', class_counts=counts)

        2. Manual vector:
           loss = CostSensitiveCE(cost_vector=[1.0, 2.0, 4.0, ...])

        3. Manual matrix:
           cost_mat = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
           loss = CostSensitiveCE(cost_matrix=cost_mat)

    Args:
        cost_vector: Manual per-class cost vector (C,)
        cost_matrix: Manual cost matrix (C, C)
        cost_strategy: 'auto' (inv_freq), 'sqrt', 'log', 'uniform', 'manual'
        class_counts: Class sample counts (required for auto strategies)
        normalize: Normalize costs to mean=1
        zero_diagonal: Zero diagonal of cost matrix
    """

    def __init__(
            self,
            cost_vector: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
            cost_matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
            cost_strategy: str = 'auto',
            class_counts: Optional[Union[Sequence[int], np.ndarray]] = None,
            normalize: bool = True,
            zero_diagonal: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._raw_cost_vector = cost_vector
        self._raw_cost_matrix = cost_matrix
        self.cost_strategy = cost_strategy
        self.class_counts = class_counts
        self.normalize = normalize
        self.zero_diagonal = zero_diagonal
        self._w_cache: Optional[torch.Tensor] = None
        self._w_device = None

        # Validate configuration
        if cost_strategy == 'manual':
            if cost_vector is None and cost_matrix is None:
                raise ValueError(
                    "cost_strategy='manual' requires cost_vector or cost_matrix. "
                    "Use cost_strategy='auto' for automatic generation."
                )
        elif cost_strategy in ['auto', 'sqrt', 'log']:
            if class_counts is None:
                raise ValueError(
                    f"cost_strategy='{cost_strategy}' requires class_counts. "
                    f"Provide class_counts or use cost_strategy='uniform'."
                )

    def _weights(self, device, num_classes: int) -> torch.Tensor:
        """Get or compute cost weights."""
        if self._w_cache is None or self._w_device != device:
            self._w_cache = _prepare_cost_vector_from_inputs(
                num_classes,
                device,
                cost_vector=self._raw_cost_vector,
                cost_matrix=self._raw_cost_matrix,
                cost_strategy=self.cost_strategy,
                class_counts=self.class_counts,
                zero_diagonal=self.zero_diagonal,
                normalize=self.normalize
            )
            self._w_device = device

            # Print info on first call
            if self.cost_strategy != 'uniform':
                print(f"[CostSensitiveCE] Strategy: {self.cost_strategy}")
                print(f"  Cost range: [{self._w_cache.min().item():.3f}, {self._w_cache.max().item():.3f}]")
                print(f"  Cost mean: {self._w_cache.mean().item():.3f}")

        return self._w_cache

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        C = logits.size(-1)
        w = self._weights(logits.device, C)  # (C,)
        return F.cross_entropy(logits, targets, weight=w, reduction=self.reduction)


class CostSensitiveExpectedLoss(BaseLoss):
    """
    Expected Cost / Bayes Risk with automatic cost generation.

    Args:
        cost_matrix: Manual cost matrix (C, C)
        cost_strategy: 'auto', 'sqrt', 'log', 'uniform', 'manual'
        class_counts: Required for auto strategies
        row_normalize: Normalize each row's off-diagonal to mean=1
        zero_diagonal: Zero diagonal of cost matrix
    """

    def __init__(
            self,
            cost_matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
            cost_strategy: str = 'auto',
            class_counts: Optional[Union[Sequence[int], np.ndarray]] = None,
            row_normalize: bool = True,
            zero_diagonal: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cost_strategy = cost_strategy
        self.class_counts = class_counts
        self.row_normalize = row_normalize
        self.zero_diagonal = zero_diagonal
        self._cm_cache: Optional[torch.Tensor] = None
        self._cm_device = None

        # For manual mode
        if cost_strategy == 'manual':
            if cost_matrix is None:
                raise ValueError("cost_strategy='manual' requires cost_matrix")
            self._raw_cost_matrix = cost_matrix
        else:
            self._raw_cost_matrix = None
            if cost_strategy in ['auto', 'sqrt', 'log'] and class_counts is None:
                raise ValueError(f"cost_strategy='{cost_strategy}' requires class_counts")

    def _prepare_matrix(self, device, num_classes: int) -> torch.Tensor:
        """Prepare cost matrix."""
        if self.cost_strategy == 'manual':
            C = TensorUtils.as_tensor2d(self._raw_cost_matrix, device=device, dtype=torch.float)
            if C.shape != (num_classes, num_classes):
                raise ValueError(f"cost_matrix must be ({num_classes},{num_classes}), got {tuple(C.shape)}")

        elif self.cost_strategy == 'uniform':
            # All off-diagonal costs = 1
            C = torch.ones(num_classes, num_classes, device=device, dtype=torch.float)

        else:  # auto, sqrt, log
            # Generate cost vector first
            counts = np.asarray(self.class_counts).astype(np.float64)
            counts = np.clip(counts, 1.0, None)

            if self.cost_strategy == 'auto':
                cost_vec = 1.0 / counts
            elif self.cost_strategy == 'sqrt':
                cost_vec = 1.0 / np.sqrt(counts)
            elif self.cost_strategy == 'log':
                cost_vec = 1.0 / np.log1p(counts)

            # Build matrix: C[i,j] = cost of misclassifying i as j
            # Use asymmetric costs: misclassifying minority is more costly
            C = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float)
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j:
                        # Cost = true class cost (penalize minority more)
                        C[i, j] = cost_vec[i]

        C = C.clone().clamp_min(0.0)
        if self.zero_diagonal:
            C.fill_diagonal_(0.0)

        if self.row_normalize:
            off_sum = C.sum(dim=1)
            denom = max(num_classes - 1, 1)
            scale = (off_sum / float(denom)).clamp_min(1e-12)
            C = C / scale.unsqueeze(1)

        return C

    def _matrix(self, device, num_classes: int) -> torch.Tensor:
        if self._cm_cache is None or self._cm_device != device:
            self._cm_cache = self._prepare_matrix(device, num_classes)
            self._cm_device = device
            print(f"[CostSensitiveExpected] Strategy: {self.cost_strategy}")
        return self._cm_cache

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        num_classes = logits.size(-1)
        cm = self._matrix(logits.device, num_classes)  # (C, C)
        probs = F.softmax(logits, dim=-1)  # (B, C)
        cost_rows = cm.index_select(0, targets)  # (B, C)
        sample_loss = (cost_rows * probs).sum(dim=1)
        return self._apply_reduction(sample_loss)


class CostSensitiveFocal(BaseLoss):
    """
    Cost-sensitive Focal Loss with automatic cost generation.

    Args:
        gamma: Focal loss focusing parameter
        cost_vector: Manual cost vector
        cost_matrix: Manual cost matrix (for extracting vector)
        cost_strategy: 'auto', 'sqrt', 'log', 'uniform', 'manual'
        class_counts: Required for auto strategies
        normalize: Normalize costs to mean=1
        zero_diagonal: Zero diagonal of cost matrix
    """

    def __init__(
            self,
            gamma: float = 2.0,
            cost_vector: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
            cost_matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]] = None,
            cost_strategy: str = 'auto',
            class_counts: Optional[Union[Sequence[int], np.ndarray]] = None,
            normalize: bool = True,
            zero_diagonal: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self._raw_cost_vector = cost_vector
        self._raw_cost_matrix = cost_matrix
        self.cost_strategy = cost_strategy
        self.class_counts = class_counts
        self.normalize = normalize
        self.zero_diagonal = zero_diagonal
        self._alpha_cache: Optional[torch.Tensor] = None
        self._alpha_device = None

        if cost_strategy == 'manual':
            if cost_vector is None and cost_matrix is None:
                raise ValueError("cost_strategy='manual' requires cost_vector or cost_matrix")
        elif cost_strategy in ['auto', 'sqrt', 'log']:
            if class_counts is None:
                raise ValueError(f"cost_strategy='{cost_strategy}' requires class_counts")

    def _alpha_vec(self, device, num_classes: int) -> torch.Tensor:
        if self._alpha_cache is None or self._alpha_device != device:
            self._alpha_cache = _prepare_cost_vector_from_inputs(
                num_classes,
                device,
                cost_vector=self._raw_cost_vector,
                cost_matrix=self._raw_cost_matrix,
                cost_strategy=self.cost_strategy,
                class_counts=self.class_counts,
                zero_diagonal=self.zero_diagonal,
                normalize=self.normalize
            )
            self._alpha_device = device
            print(f"[CostSensitiveFocal] Strategy: {self.cost_strategy}, Gamma: {self.gamma}")
        return self._alpha_cache

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        C = logits.size(-1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = torch.pow(1 - pt, self.gamma)

        alpha_vec = self._alpha_vec(logits.device, C)  # (C,)
        alpha_t = alpha_vec.gather(0, targets)  # (B,)
        loss = alpha_t * focal_weight * ce_loss
        return self._apply_reduction(loss)

# =============================================================================
# Advanced and Composite Losses
# =============================================================================

class LabelSmoothingLoss(BaseLoss):
    """Label smoothing with class-aware smoothing rates."""

    def __init__(
        self,
        smoothing: float = 0.1,
        class_counts: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
        adaptive: bool = False,
        adaptive_power: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.smoothing = float(smoothing)
        self.class_counts = class_counts
        self.adaptive = bool(adaptive)
        self.adaptive_power = float(adaptive_power)
        self._smoothing_rates: Optional[torch.Tensor] = None

    def _compute_smoothing_rates(self, device) -> torch.Tensor:
        """Compute per-class smoothing rates."""
        if not self.adaptive or self.class_counts is None:
            # Uniform smoothing
            num_classes = len(self.class_counts) if self.class_counts is not None else 1
            return torch.full((num_classes,), self.smoothing, device=device)

        if self._smoothing_rates is None:
            counts = TensorUtils.as_tensor1d(self.class_counts, device=device, dtype=torch.float)
            max_count = counts.max().clamp_min(1.0)
            # Smaller classes get less smoothing
            rates = self.smoothing * torch.pow(counts / max_count, self.adaptive_power)
            self._smoothing_rates = rates.clamp(0.0, 0.9)

        return self._smoothing_rates.to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        targets = TensorUtils.to_1d_long(targets)
        num_classes = logits.size(-1)

        if self.adaptive and self.class_counts is not None:
            smoothing_rates = self._compute_smoothing_rates(logits.device)
            if smoothing_rates.numel() != num_classes:
                raise ValueError(
                    f"LabelSmoothingLoss(adaptive): len(class_counts)={smoothing_rates.numel()} "
                    f"!= num_classes={num_classes}"
                )

            smoothing_per_sample = smoothing_rates.gather(0, targets)

            # Smooth labels
            true_dist = TensorUtils.one_hot(targets, num_classes)
            smooth_dist = (1 - smoothing_per_sample.unsqueeze(1)) * true_dist + \
                          (smoothing_per_sample / (num_classes - 1)).unsqueeze(1) * (1 - true_dist)

            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_dist * log_probs).sum(dim=-1)
            return self._apply_reduction(loss)
        else:
            # Standard uniform smoothing
            return F.cross_entropy(
                logits, targets,
                label_smoothing=self.smoothing,
                reduction=self.reduction
            )


class CombinedLoss(BaseLoss):
    """Combine multiple losses with weights."""

    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}

        # Validate weights
        if set(self.weights.keys()) != set(losses.keys()):
            raise ValueError("Loss weights must match loss names")

    def update_epoch(self, epoch: int):
        """Update all component losses."""
        for loss in self.losses.values():
            if hasattr(loss, 'update_epoch'):
                loss.update_epoch(epoch)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        total_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(logits, targets, **kwargs)
            total_loss = total_loss + self.weights[name] * component_loss
        return total_loss


# =============================================================================
# Knowledge Distillation Losses
# =============================================================================

class KnowledgeDistillationLoss(BaseLoss):
    """Knowledge Distillation with temperature scaling."""

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        temperature: float = 3.0,
        alpha: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_loss = base_loss or CrossEntropy()
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        teacher_logits = kwargs.get('teacher_logits')
        if teacher_logits is None:
            raise ValueError("KnowledgeDistillationLoss requires 'teacher_logits' in kwargs")

        # Hard target loss
        hard_loss = self.base_loss(logits, targets)

        # Soft target loss
        student_soft = F.log_softmax(logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)

        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss


# =============================================================================
# Loss Factory and Registry
# =============================================================================

LOSS_REGISTRY = {
    # Basic losses
    'CrossEntropy': CrossEntropy,
    'FocalLoss': FocalLoss,

    # Reweighting losses
    'ClassBalancedLoss': ClassBalancedLoss,
    'LDAMLoss': LDAMLoss,
    'ProgressiveLoss': ProgressiveLoss,

    # Prior adjustment losses
    'BalancedSoftmaxLoss': BalancedSoftmaxLoss,
    'LogitAdjustmentLoss': LogitAdjustmentLoss,

    # Cost-sensitive losses
    'CostSensitiveCE': CostSensitiveCE,
    'CostSensitiveExpected': CostSensitiveExpectedLoss,
    'CostSensitiveFocal': CostSensitiveFocal,

    # Advanced losses
    'LabelSmoothingLoss': LabelSmoothingLoss,
    'CombinedLoss': CombinedLoss,
    'KnowledgeDistillationLoss': KnowledgeDistillationLoss,
}

# Losses that require class counts
LOSSES_REQUIRING_COUNTS = {
    'ClassBalancedLoss',
    'LDAMLoss',
    'BalancedSoftmaxLoss',
    'LogitAdjustmentLoss',
    'ProgressiveLoss',
}


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_name: Name of the loss function
        **kwargs: Arguments to pass to the loss constructor

    Returns:
        Instantiated loss function
    """
    if loss_name not in LOSS_REGISTRY:
        available = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {available}")

    loss_class = LOSS_REGISTRY[loss_name]
    return loss_class(**kwargs)


def get_loss_info() -> Dict[str, Dict[str, str]]:
    """Get information about available loss functions."""
    return {
        'Basic Losses': {
            'CrossEntropy': 'Standard cross-entropy with optional label smoothing',
            'FocalLoss': 'Focal loss for addressing class imbalance',
        },
        'Reweighting Losses': {
            'ClassBalancedLoss': 'Class-balanced loss using effective number of samples',
            'LDAMLoss': 'Label-distribution-aware margin loss with DRW',
            'ProgressiveLoss': 'Progressive reweighting over training epochs',
        },
        'Prior Adjustment': {
            'BalancedSoftmaxLoss': 'Balanced softmax with prior adjustment',
            'LogitAdjustmentLoss': 'Logit adjustment with tunable prior strength',
        },
        'Cost-Sensitive': {
            'CostSensitiveCE': 'Cost-weighted cross-entropy using per-class cost vector',
            'CostSensitiveExpected': 'Expected cost / Bayes risk over softmax probabilities',
            'CostSensitiveFocal': 'Focal loss with cost-derived per-class alpha',
        },
        'Advanced Losses': {
            'LabelSmoothingLoss': 'Label smoothing with optional class-aware rates',
            'CombinedLoss': 'Combination of multiple loss functions',
            'KnowledgeDistillationLoss': 'Knowledge distillation with temperature scaling',
        }
    }


def list_available_losses() -> List[str]:
    """List all available loss functions."""
    return sorted(list(LOSS_REGISTRY.keys()))


def requires_class_counts(loss_name: str) -> bool:
    """Check if a loss function requires class counts."""
    return loss_name in LOSSES_REQUIRING_COUNTS


# =============================================================================
# Usage Examples and Utilities
# =============================================================================

class LossScheduler:
    """Scheduler for changing loss functions or parameters during training."""

    def __init__(self, loss_configs: Dict[int, Dict]):
        """
        Args:
            loss_configs: Dict mapping epoch to loss configuration
                          {epoch: {'loss_name': str, 'params': dict}}
        """
        self.loss_configs = loss_configs
        self.current_loss: Optional[nn.Module] = None
        self.current_epoch = -1

    def get_loss(self, epoch: int) -> nn.Module:
        """Get loss function for given epoch."""
        applicable_epoch = max([e for e in self.loss_configs.keys() if e <= epoch], default=0)

        if applicable_epoch != self.current_epoch:
            config = self.loss_configs[applicable_epoch]
            self.current_loss = create_loss(config['loss_name'], **config.get('params', {}))
            self.current_epoch = applicable_epoch

        # Update epoch for progressive/epoch-aware losses
        if hasattr(self.current_loss, 'update_epoch'):
            self.current_loss.update_epoch(epoch)

        return self.current_loss


def validate_loss_setup(
    loss: nn.Module,
    class_counts: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
    num_classes: Optional[int] = None
) -> bool:
    """
    Validate loss function setup.

    Args:
        loss: Loss function to validate
        class_counts: Class counts (if available)
        num_classes: Number of classes (if available)

    Returns:
        True if setup is valid
    """
    loss_name = loss.__class__.__name__

    # Check if loss requires class counts
    if requires_class_counts(loss_name):
        if class_counts is None:
            raise ValueError(f"{loss_name} requires class_counts but none provided")

        if num_classes is not None:
            if len(class_counts) != num_classes:
                raise ValueError(
                    f"Length of class_counts ({len(class_counts)}) doesn't match num_classes ({num_classes})"
                )

    # Additional validations can be added here
    return True


"""
Usage Examples:

# === Cost-Sensitive (provide either cost_vector or cost_matrix) ===

# 1) Cost-weighted CE with cost vector (higher means更重视此类误分代价)
C = 5
cost_vec = [1.0, 2.0, 4.0, 1.5, 0.8]
loss = create_loss('CostSensitiveCE', cost_vector=cost_vec)

# 2) Cost-weighted CE 从代价矩阵推导（每行的非对角平均作为该类代价）
cost_mat = np.ones((C, C), dtype=np.float32)
np.fill_diagonal(cost_mat, 0.0)  # 正确分类代价为0
# 例如可把把第2类→其它的代价设为更大：
cost_mat[2, :] *= 3.0; cost_mat[2, 2] = 0.0
loss = create_loss('CostSensitiveCE', cost_matrix=cost_mat, normalize=True)

# 3) Expected Cost（贝叶斯风险），直接最小化 E_C[Cost(y, j)]
loss = create_loss('CostSensitiveExpected', cost_matrix=cost_mat, row_normalize=True)

# 4) Cost-Sensitive Focal：用代价值构造 alpha_c，配合 gamma
loss = create_loss('CostSensitiveFocal', cost_matrix=cost_mat, gamma=2.0)

# === 其余示例保留 ===

1. Basic cross-entropy:
   loss = create_loss('CrossEntropy', label_smoothing=0.1)

2. Focal loss:
   loss = create_loss('FocalLoss', gamma=2.0, alpha=0.25)

3. Class-balanced loss:
   class_counts = [1000, 100, 50]
   loss = create_loss('ClassBalancedLoss', class_counts=class_counts, beta=0.9999)

4. LDAM loss with DRW:
   loss = create_loss('LDAMLoss',
                      class_counts=class_counts,
                      max_margin=0.5,
                      drw_start_epoch=160)

5. Progressive loss:
   loss = create_loss('ProgressiveLoss',
                      class_counts=class_counts,
                      total_epochs=200,
                      start_strategy='uniform',
                      end_strategy='inverse')

6. Balanced softmax / Logit adjustment (二选一，不要叠加):
   loss = create_loss('BalancedSoftmaxLoss', class_counts=class_counts)
   # or
   loss = create_loss('LogitAdjustmentLoss', class_counts=class_counts, tau=1.2)

7. Combined loss:
   losses = {
       'ce': create_loss('CrossEntropy'),
       'focal': create_loss('FocalLoss', gamma=2.0)
   }
   weights = {'ce': 0.7, 'focal': 0.3}
   loss = create_loss('CombinedLoss', losses=losses, weights=weights)

8. Loss scheduling:
   scheduler = LossScheduler({
       0:   {'loss_name': 'CrossEntropy', 'params': {}},
       50:  {'loss_name': 'FocalLoss', 'params': {'gamma': 2.0}},
       100: {'loss_name': 'LDAMLoss', 'params': {'class_counts': class_counts}}
   })

   # During training
   for epoch in range(200):
       loss_fn = scheduler.get_loss(epoch)
       # use loss_fn(...)
"""
