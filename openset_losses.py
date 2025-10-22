# -*- coding: utf-8 -*-
"""
Loss Functions for Long-Tail Open-Set Recognition

This module combines long-tail learning losses with open-set objectives:
1. Closed-set classification on known classes (with long-tail handling)
2. Open-set detection via various approaches
3. Feature diffusion regularization
4. Contrastive learning for better feature separation

Key Innovations:
- Joint optimization of long-tail classification + open-set detection
- Diffusion-based feature regularization
- Class-balanced contrastive loss for tail classes
- Entropy regularization for confident predictions

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import base losses
from imbalanced_losses import (
    FocalLoss,
    LDAMLoss,
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
)


# =============================================================================
# Entropy Regularization for Open-Set
# =============================================================================

class EntropyLoss(nn.Module):
    """
    Entropy regularization loss.

    Encourages confident predictions for known classes (low entropy)
    and uncertain predictions for potential unknown samples (high entropy).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of predictions.

        Args:
            logits: Model logits [B, C]

        Returns:
            entropy: Entropy values [B] or scalar
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)

        if self.reduction == "mean":
            return entropy.mean()
        elif self.reduction == "sum":
            return entropy.sum()
        else:
            return entropy


# =============================================================================
# Objectosphere Loss (for Open-Set)
# =============================================================================

class ObjectosphereLoss(nn.Module):
    """
    Objectosphere Loss for open-set recognition.

    Creates a "sphere" around each class in feature space. Samples outside
    all spheres are considered unknown.

    Reference: "Reducing Network Agnostophobia" (NeurIPS 2018)
    """

    def __init__(self, num_classes: int, feature_dim: int, xi: float = 0.1):
        """
        Args:
            num_classes: Number of known classes
            feature_dim: Feature dimension
            xi: Radius of unknownness
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.xi = xi

        # Class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute objectosphere loss.

        Args:
            features: Feature vectors [B, D]
            labels: Class labels [B]

        Returns:
            loss: Objectosphere loss
        """
        batch_size = features.shape[0]

        # Compute distances to all centers
        # [B, C] = ||features||^2 + ||centers||^2 - 2 * features @ centers^T
        feat_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # [B, 1]
        cent_sq = torch.sum(self.centers ** 2, dim=1, keepdim=True).t()  # [1, C]
        distances = feat_sq + cent_sq - 2 * features @ self.centers.t()  # [B, C]

        # For each sample, get distance to its class center
        labels_clamped = torch.clamp(labels, 0, self.num_classes - 1)
        distances_to_gt = distances[torch.arange(batch_size), labels_clamped]

        # Loss: push samples inside their class sphere
        # loss = max(0, distance_to_gt - xi)
        loss = F.relu(distances_to_gt - self.xi ** 2)

        return loss.mean()


# =============================================================================
# Contrastive Loss for Feature Learning
# =============================================================================

class ClassBalancedContrastiveLoss(nn.Module):
    """
    Class-balanced supervised contrastive loss for long-tail scenarios.

    Samples from the same class are pulled together, samples from different
    classes are pushed apart. Uses class-balanced sampling to avoid head class
    dominance.

    Reference: "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        class_counts: Optional[np.ndarray] = None,
        use_reweight: bool = True,
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            class_counts: Sample counts per class (for reweighting)
            use_reweight: Whether to use class-balanced reweighting
        """
        super().__init__()
        self.temperature = temperature
        self.use_reweight = use_reweight

        if class_counts is not None and use_reweight:
            # Inverse frequency weighting
            weights = 1.0 / (class_counts + 1e-6)
            weights = weights / weights.sum()
            self.register_buffer("class_weights", torch.from_numpy(weights).float())
        else:
            self.class_weights = None

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced contrastive loss.

        Args:
            features: L2-normalized features [B, D]
            labels: Class labels [B]

        Returns:
            loss: Contrastive loss
        """
        batch_size = features.shape[0]
        device = features.device

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature  # [B, B]

        # Mask for positive pairs (same class, different samples)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).float().to(device)
        mask_pos.fill_diagonal_(0)  # Exclude self

        # Mask for negative pairs (different class)
        mask_neg = (labels != labels.t()).float().to(device)

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log(sum(exp(logits))) for negatives and self
        exp_logits = torch.exp(logits) * mask_neg + torch.exp(logits).diagonal().diag()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive pairs
        num_pos_per_sample = mask_pos.sum(dim=1)
        valid_samples = num_pos_per_sample > 0

        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (num_pos_per_sample + 1e-6)

        # Apply class weights if available
        if self.class_weights is not None:
            # Ensure the cached weights live on the same device as the features
            weights = self.class_weights.to(device)
            label_indices = labels.view(-1).to(torch.long)
            weights = weights[label_indices]
            mean_log_prob_pos = mean_log_prob_pos * weights

        # Loss
        loss = -mean_log_prob_pos[valid_samples].mean()

        return loss


# =============================================================================
# Joint Loss for Long-Tail Open-Set Recognition
# =============================================================================

class LongTailOpenSetLoss(nn.Module):
    """
    Joint loss function for long-tail open-set recognition.

    Combines:
    1. Closed-set classification loss (with long-tail handling)
    2. Feature diffusion loss (for open-set detection)
    3. Contrastive loss (for better feature separation)
    4. Entropy regularization (for confident predictions)
    5. Objectosphere loss (optional, for explicit open-set modeling)

    Args:
        num_classes: Number of known classes
        class_counts: Sample counts per class
        loss_type: Base classification loss ("ce", "focal", "ldam", "balanced_softmax", "cb")
        use_diffusion: Whether to use diffusion regularization
        use_contrastive: Whether to use contrastive loss
        use_entropy: Whether to use entropy regularization
        use_objectosphere: Whether to use objectosphere loss
        feature_dim: Feature dimension (required for objectosphere)
        lambda_diffusion: Weight for diffusion loss
        lambda_contrastive: Weight for contrastive loss
        lambda_entropy: Weight for entropy loss
        lambda_objectosphere: Weight for objectosphere loss
        **loss_kwargs: Additional arguments for base loss
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: np.ndarray,
        loss_type: str = "balanced_softmax",
        use_diffusion: bool = True,
        use_contrastive: bool = True,
        use_entropy: bool = False,
        use_objectosphere: bool = False,
        feature_dim: Optional[int] = None,
        lambda_diffusion: float = 0.1,
        lambda_contrastive: float = 0.1,
        lambda_entropy: float = 0.01,
        lambda_objectosphere: float = 0.1,
        **loss_kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_diffusion = use_diffusion
        self.use_contrastive = use_contrastive
        self.use_entropy = use_entropy
        self.use_objectosphere = use_objectosphere

        self.lambda_diffusion = lambda_diffusion
        self.lambda_contrastive = lambda_contrastive
        self.lambda_entropy = lambda_entropy
        self.lambda_objectosphere = lambda_objectosphere

        # Base classification loss
        if loss_type == "ce":
            self.classification_loss = nn.CrossEntropyLoss()
        elif loss_type == "focal":
            self.classification_loss = FocalLoss(num_classes=num_classes, **loss_kwargs)
        elif loss_type == "ldam":
            self.classification_loss = LDAMLoss(
                num_classes=num_classes,
                class_counts=class_counts,
                **loss_kwargs
            )
        elif loss_type == "balanced_softmax":
            self.classification_loss = BalancedSoftmaxLoss(
                class_counts=class_counts,
                **loss_kwargs
            )
        elif loss_type == "cb":
            self.classification_loss = ClassBalancedLoss(
                num_classes=num_classes,
                class_counts=class_counts,
                **loss_kwargs
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Contrastive loss
        if use_contrastive:
            self.contrastive_loss = ClassBalancedContrastiveLoss(
                class_counts=class_counts,
                temperature=0.07,
            )

        # Entropy loss
        if use_entropy:
            self.entropy_loss = EntropyLoss()

        # Objectosphere loss
        if use_objectosphere:
            if feature_dim is None:
                raise ValueError("feature_dim must be specified when use_objectosphere=True")
            self.objectosphere_loss = ObjectosphereLoss(
                num_classes=num_classes,
                feature_dim=feature_dim,
            )

        print(f"[LongTailOpenSetLoss] Initialized:")
        print(f"  - Base loss: {loss_type}")
        print(f"  - Use diffusion: {use_diffusion} (lambda={lambda_diffusion})")
        print(f"  - Use contrastive: {use_contrastive} (lambda={lambda_contrastive})")
        print(f"  - Use entropy: {use_entropy} (lambda={lambda_entropy})")
        print(f"  - Use objectosphere: {use_objectosphere} (lambda={lambda_objectosphere})")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        diffusion_model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss.

        Args:
            logits: Model logits [B, C]
            labels: Ground truth labels [B]
            features: Feature vectors [B, D] (required for contrastive/objectosphere/diffusion)
            diffusion_model: Diffusion model (required if use_diffusion=True)

        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}

        # 1. Classification loss
        cls_loss = self.classification_loss(logits, labels)
        loss_dict["cls"] = cls_loss.item()
        total_loss = cls_loss

        # 2. Diffusion loss
        if self.use_diffusion:
            if diffusion_model is None or features is None:
                raise ValueError("diffusion_model and features must be provided when use_diffusion=True")

            diff_loss, _ = diffusion_model(features, labels)
            loss_dict["diffusion"] = diff_loss.item()
            total_loss = total_loss + self.lambda_diffusion * diff_loss

        # 3. Contrastive loss
        if self.use_contrastive:
            if features is None:
                raise ValueError("features must be provided when use_contrastive=True")

            con_loss = self.contrastive_loss(features, labels)
            loss_dict["contrastive"] = con_loss.item()
            total_loss = total_loss + self.lambda_contrastive * con_loss

        # 4. Entropy loss
        if self.use_entropy:
            ent_loss = self.entropy_loss(logits)
            loss_dict["entropy"] = ent_loss.item()
            total_loss = total_loss - self.lambda_entropy * ent_loss  # Minimize entropy

        # 5. Objectosphere loss
        if self.use_objectosphere:
            if features is None:
                raise ValueError("features must be provided when use_objectosphere=True")

            obj_loss = self.objectosphere_loss(features, labels)
            loss_dict["objectosphere"] = obj_loss.item()
            total_loss = total_loss + self.lambda_objectosphere * obj_loss

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


# =============================================================================
# Factory Function
# =============================================================================

def create_longtail_openset_loss(
    num_classes: int,
    class_counts: np.ndarray,
    loss_config: Dict[str, Any],
) -> LongTailOpenSetLoss:
    """
    Factory function to create joint long-tail open-set loss.

    Args:
        num_classes: Number of known classes
        class_counts: Sample counts per class
        loss_config: Loss configuration dictionary

    Returns:
        Joint loss function
    """
    return LongTailOpenSetLoss(
        num_classes=num_classes,
        class_counts=class_counts,
        **loss_config,
    )
