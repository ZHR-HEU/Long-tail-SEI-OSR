# -*- coding: utf-8 -*-
"""
Open-Set Recognition Methods for Long-Tail Scenarios

This module implements various open-set recognition techniques:
1. OpenMax - Uses extreme value theory (EVT) to model tail distributions
2. ODIN - Out-of-Distribution detector using temperature scaling and input perturbation
3. Energy-based OOD - Uses free energy as uncertainty measure
4. Maximum Softmax Probability (MSP) - Baseline method
5. Mahalanobis Distance - Fits Gaussian to each class in feature space
6. Diffusion-based - Uses reconstruction error from diffusion models

These methods are adapted to handle long-tail distributions where tail classes
have limited training samples.

References:
- OpenMax: "Towards Open Set Deep Networks" (CVPR 2016)
- ODIN: "Enhancing The Reliability of Out-of-distribution Image Detection" (ICLR 2018)
- Energy-based: "Energy-based Out-of-distribution Detection" (NeurIPS 2020)
- Mahalanobis: "A Simple Unified Framework for Detecting OOD" (NeurIPS 2018)

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import weibull_min


# =============================================================================
# Data Classes for Storing Open-Set Models
# =============================================================================

@dataclass
class OpenMaxModel:
    """Stores OpenMax model parameters."""
    means: np.ndarray  # Class means in activation space [num_classes, feature_dim]
    weibull_models: List[Dict[str, Any]]  # Weibull models for each class
    num_classes: int
    tailsize: int = 20  # Number of top samples used to fit Weibull


@dataclass
class MahalanobisModel:
    """Stores Mahalanobis distance model parameters."""
    class_means: torch.Tensor  # [num_classes, feature_dim]
    precision: torch.Tensor  # Shared precision matrix [feature_dim, feature_dim]
    num_classes: int


# =============================================================================
# Maximum Softmax Probability (MSP) - Baseline
# =============================================================================

class MaxSoftmaxProb:
    """
    Baseline open-set detector using maximum softmax probability.

    Simple but effective baseline: use max(softmax(logits)) as confidence.
    Samples with low max probability are classified as unknown.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Confidence threshold for known/unknown classification
        """
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict known/unknown based on max softmax probability.

        Args:
            logits: Model logits [B, num_classes]

        Returns:
            scores: Confidence scores [B] (higher = more confident in known class)
            predictions: Class predictions [B] (or -1 for unknown)
        """
        probs = F.softmax(logits, dim=1)
        scores, predictions = torch.max(probs, dim=1)

        # Mark low-confidence predictions as unknown (-1)
        predictions = torch.where(scores >= self.threshold, predictions, torch.tensor(-1, device=predictions.device))

        return scores, predictions


# =============================================================================
# ODIN (Out-of-DIstribution detector for Neural networks)
# =============================================================================

class ODIN:
    """
    ODIN: Temperature scaling + input perturbation for OOD detection.

    Uses two techniques:
    1. Temperature scaling: softmax(logits / T) to separate ID/OOD
    2. Input perturbation: Add small gradient-based noise to inputs
    """

    def __init__(self, temperature: float = 1000.0, epsilon: float = 0.0012, threshold: float = 0.5):
        """
        Args:
            temperature: Temperature for scaling (higher = more separation)
            epsilon: Perturbation magnitude
            threshold: Confidence threshold
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.threshold = threshold

    def predict(
        self,
        model: nn.Module,
        features: torch.Tensor,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using ODIN with input perturbation.

        Args:
            model: Classification model (should output logits given features)
            features: Input features [B, D]
            requires_grad: Whether to apply input perturbation

        Returns:
            scores: ODIN scores [B] (higher = more confident in known)
            predictions: Class predictions [B] (or -1 for unknown)
        """
        if requires_grad:
            features = features.detach().clone().requires_grad_(True)

            # Forward pass
            logits = model(features)

            # Temperature scaling
            scaled_logits = logits / self.temperature

            # Get max softmax value
            max_scores = F.softmax(scaled_logits, dim=1).max(dim=1)[0]

            # Compute gradients for perturbation
            loss = max_scores.sum()
            loss.backward()

            # Add perturbation in gradient direction
            gradient = features.grad.data
            features_perturbed = features - self.epsilon * torch.sign(gradient)

            # Re-compute logits with perturbed input
            with torch.no_grad():
                logits = model(features_perturbed)
        else:
            with torch.no_grad():
                logits = model(features)

        # Temperature scaling
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        scores, predictions = torch.max(probs, dim=1)

        # Mark unknown
        predictions = torch.where(scores >= self.threshold, predictions, torch.tensor(-1, device=predictions.device))

        return scores, predictions


# =============================================================================
# Energy-Based Out-of-Distribution Detection
# =============================================================================

class EnergyBasedOOD:
    """
    Energy-based OOD detection.

    Uses free energy E(x) = -log(sum(exp(f_i(x)))) as uncertainty measure.
    Lower energy indicates higher confidence in known classes.

    Reference: "Energy-based Out-of-distribution Detection" (NeurIPS 2020)
    """

    def __init__(self, threshold: float = -10.0, temperature: float = 1.0):
        """
        Args:
            threshold: Energy threshold (lower = more confident)
            temperature: Temperature scaling
        """
        self.threshold = threshold
        self.temperature = temperature

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using energy scores.

        Args:
            logits: Model logits [B, num_classes]

        Returns:
            scores: Energy scores [B] (lower energy = more confident in known)
            predictions: Class predictions [B] (or -1 for unknown)
        """
        # Compute energy: E(x) = -T * log(sum(exp(f_i(x) / T)))
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

        # Get class predictions
        predictions = torch.argmax(logits, dim=1)

        # Mark unknown (high energy)
        predictions = torch.where(energy <= self.threshold, predictions, torch.tensor(-1, device=predictions.device))

        # Return negative energy as score (higher = more confident)
        return -energy, predictions


# =============================================================================
# OpenMax - EVT-based Open-Set Recognition
# =============================================================================

class OpenMax:
    """
    OpenMax: Open-Set recognition using Extreme Value Theory.

    Fits Weibull distributions to activation vectors of correctly classified
    training samples, then uses these to estimate probability of unknown class.

    Reference: "Towards Open Set Deep Networks" (CVPR 2016)
    """

    def __init__(self, alpha: int = 10, tailsize: int = 20):
        """
        Args:
            alpha: Number of top classes to revise (typically 10)
            tailsize: Number of samples to use for fitting Weibull
        """
        self.alpha = alpha
        self.tailsize = tailsize
        self.model: Optional[OpenMaxModel] = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        logits: np.ndarray,
        num_classes: int,
    ):
        """
        Fit OpenMax model using training data.

        Args:
            features: Training features [N, D]
            labels: Training labels [N]
            logits: Training logits [N, num_classes]
            num_classes: Number of classes
        """
        print(f"[OpenMax] Fitting Weibull models for {num_classes} classes...")

        # Compute class means (MAV - Mean Activation Vector)
        class_means = np.zeros((num_classes, features.shape[1]))
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_means[c] = features[mask].mean(axis=0)

        # Fit Weibull distribution for each class
        weibull_models = []
        for c in range(num_classes):
            # Get correctly classified samples for class c
            mask = (labels == c) & (logits.argmax(axis=1) == c)
            if mask.sum() < self.tailsize:
                warnings.warn(f"Class {c} has only {mask.sum()} correct samples, less than tailsize={self.tailsize}")

            class_features = features[mask]

            if len(class_features) == 0:
                # No samples - use dummy Weibull
                weibull_models.append({"shape": 1.0, "scale": 1.0, "mean": class_means[c]})
                continue

            # Compute distances to class mean
            distances = np.linalg.norm(class_features - class_means[c], axis=1)

            # Use top tailsize distances for fitting
            if len(distances) > self.tailsize:
                distances = np.sort(distances)[-self.tailsize:]

            # Fit Weibull distribution
            if len(distances) >= 2:
                shape, loc, scale = weibull_min.fit(distances, floc=0)
            else:
                shape, scale = 1.0, 1.0

            weibull_models.append({
                "shape": shape,
                "scale": scale,
                "mean": class_means[c],
            })

        self.model = OpenMaxModel(
            means=class_means,
            weibull_models=weibull_models,
            num_classes=num_classes,
            tailsize=self.tailsize,
        )

        print(f"[OpenMax] Fitted {num_classes} Weibull models")

    @torch.no_grad()
    def predict(self, features: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using OpenMax.

        Args:
            features: Test features [B, D]
            logits: Test logits [B, num_classes]

        Returns:
            scores: OpenMax confidence scores [B]
            predictions: Class predictions [B] (or -1 for unknown)
        """
        if self.model is None:
            raise ValueError("OpenMax model not fitted. Call fit() first.")

        features_np = features.cpu().numpy()
        logits_np = logits.cpu().numpy()

        batch_size = features_np.shape[0]
        openmax_scores = np.zeros(batch_size)
        predictions_np = np.zeros(batch_size, dtype=np.int64)

        for i in range(batch_size):
            feat = features_np[i]
            logit = logits_np[i]

            # Compute distances to all class means
            distances = np.linalg.norm(self.model.means - feat, axis=1)

            # Get top-alpha classes
            ranked_indices = np.argsort(logit)[::-1][:self.alpha]

            # Compute Weibull CDF for each top class
            revised_logits = logit.copy()
            unknown_score = 0.0

            for c in ranked_indices:
                weibull = self.model.weibull_models[c]
                dist = distances[c]

                # Compute Weibull CDF (probability that distance is this extreme)
                w_score = weibull_min.cdf(dist, weibull["shape"], loc=0, scale=weibull["scale"])

                # Revise logit: reduce by w_score (transfer to unknown)
                revision = logit[c] * w_score
                revised_logits[c] -= revision
                unknown_score += revision

            # Normalize
            total = revised_logits.sum() + unknown_score
            if total > 0:
                revised_probs = revised_logits / total
                unknown_prob = unknown_score / total
            else:
                revised_probs = revised_logits
                unknown_prob = 0.0

            # Predict
            max_prob = revised_probs.max()
            pred_class = revised_probs.argmax()

            if unknown_prob > max_prob:
                predictions_np[i] = -1  # Unknown
                openmax_scores[i] = unknown_prob
            else:
                predictions_np[i] = pred_class
                openmax_scores[i] = max_prob

        scores = torch.from_numpy(openmax_scores).to(features.device)
        predictions = torch.from_numpy(predictions_np).to(features.device)

        return scores, predictions


# =============================================================================
# Mahalanobis Distance-Based OOD Detection
# =============================================================================

class MahalanobisOOD:
    """
    Mahalanobis distance-based OOD detection.

    Fits Gaussian distribution to each class in feature space,
    then uses Mahalanobis distance as confidence measure.

    Reference: "A Simple Unified Framework for Detecting OOD" (NeurIPS 2018)
    """

    def __init__(self, threshold: float = 10.0):
        """
        Args:
            threshold: Mahalanobis distance threshold
        """
        self.threshold = threshold
        self.model: Optional[MahalanobisModel] = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """
        Fit Gaussian model for each class.

        Args:
            features: Training features [N, D]
            labels: Training labels [N]
            num_classes: Number of classes
        """
        print(f"[Mahalanobis] Fitting Gaussian models for {num_classes} classes...")

        device = features.device
        feature_dim = features.shape[1]

        # Compute class means
        class_means = torch.zeros(num_classes, feature_dim, device=device)
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_means[c] = features[mask].mean(dim=0)

        # Compute tied covariance matrix
        centered_features = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centered = features[mask] - class_means[c]
                centered_features.append(centered)

        if len(centered_features) > 0:
            centered_features = torch.cat(centered_features, dim=0)
            covariance = (centered_features.T @ centered_features) / len(centered_features)

            # Add regularization
            covariance += torch.eye(feature_dim, device=device) * 1e-4

            # Compute precision matrix
            precision = torch.linalg.inv(covariance)
        else:
            precision = torch.eye(feature_dim, device=device)

        self.model = MahalanobisModel(
            class_means=class_means,
            precision=precision,
            num_classes=num_classes,
        )

        print(f"[Mahalanobis] Fitted models for {num_classes} classes")

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using Mahalanobis distance.

        Args:
            features: Test features [B, D]

        Returns:
            scores: Negative Mahalanobis distances [B] (higher = more confident)
            predictions: Class predictions [B] (or -1 for unknown)
        """
        if self.model is None:
            raise ValueError("Mahalanobis model not fitted. Call fit() first.")

        batch_size = features.shape[0]
        num_classes = self.model.num_classes

        # Compute Mahalanobis distance to each class
        min_distances = torch.zeros(batch_size, device=features.device)
        predictions = torch.zeros(batch_size, dtype=torch.long, device=features.device)

        for c in range(num_classes):
            # Compute distance: (x - mu)^T Sigma^{-1} (x - mu)
            centered = features - self.model.class_means[c]
            distances = torch.sum(centered @ self.model.precision * centered, dim=1)

            if c == 0:
                min_distances = distances
                predictions = torch.zeros_like(predictions)
            else:
                mask = distances < min_distances
                min_distances = torch.where(mask, distances, min_distances)
                predictions = torch.where(mask, torch.tensor(c, device=predictions.device), predictions)

        # Mark unknown (large distance)
        predictions = torch.where(min_distances <= self.threshold, predictions, torch.tensor(-1, device=predictions.device))

        # Return negative distance as score (higher = more confident)
        return -min_distances, predictions


# =============================================================================
# Factory Function
# =============================================================================

def create_openset_detector(
    method: str,
    num_classes: int,
    **kwargs,
) -> Any:
    """
    Factory function to create open-set detector.

    Args:
        method: "msp", "odin", "energy", "openmax", "mahalanobis"
        num_classes: Number of known classes
        **kwargs: Method-specific parameters

    Returns:
        Open-set detector instance
    """
    if method == "msp":
        return MaxSoftmaxProb(**kwargs)
    elif method == "odin":
        return ODIN(**kwargs)
    elif method == "energy":
        return EnergyBasedOOD(**kwargs)
    elif method == "openmax":
        return OpenMax(**kwargs)
    elif method == "mahalanobis":
        return MahalanobisOOD(**kwargs)
    else:
        raise ValueError(f"Unknown open-set method: {method}")
