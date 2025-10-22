# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Open-Set Recognition

This module implements various metrics for evaluating open-set recognition:
1. AUROC (Area Under ROC) - Standard OOD detection metric
2. AUPR (Area Under Precision-Recall)
3. FPR95 - False Positive Rate at 95% True Positive Rate
4. OSCR - Open-Set Classification Rate
5. Accuracy metrics (closed-set, open-set, overall)
6. Per-class metrics for long-tail analysis

Key Features:
- Handles long-tail distributions (per-group metrics)
- Combines closed-set accuracy with open-set detection
- Provides comprehensive evaluation for research

References:
- "Towards Open Set Deep Networks" (CVPR 2016)
- "Open Set Learning with Counterfactual Images" (ECCV 2020)

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)


# =============================================================================
# Data Class for Results
# =============================================================================

@dataclass
class OpenSetMetrics:
    """Container for open-set recognition metrics."""

    # Closed-set metrics (on known classes only)
    closed_set_accuracy: float
    closed_set_per_class_acc: np.ndarray

    # Open-set detection metrics
    auroc: float
    aupr: float
    fpr95: float

    # Open-set classification metrics
    oscr: float  # Open-Set Classification Rate
    open_set_f1: float

    # Overall metrics
    overall_accuracy: float

    # Long-tail specific metrics
    many_shot_acc: float
    medium_shot_acc: float
    few_shot_acc: float

    # Additional info
    num_known_classes: int
    num_test_samples: int
    num_unknown_samples: int


# =============================================================================
# Core Metric Computation Functions
# =============================================================================

def compute_auroc_aupr(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute AUROC and AUPR for open-set detection.

    Args:
        scores: Anomaly/uncertainty scores [N] (higher = more likely unknown)
        labels: Binary labels [N] (1 = unknown, 0 = known)

    Returns:
        auroc: Area under ROC curve
        aupr: Area under precision-recall curve
    """
    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    return auroc, aupr


def compute_fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute FPR at 95% TPR (common OOD detection metric).

    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Binary labels (1 = unknown, 0 = known)

    Returns:
        fpr95: False positive rate at 95% true positive rate
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(labels, scores)

    # Find FPR at TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0

    return fpr[idx[0]]


def compute_oscr(
    predictions: np.ndarray,
    scores: np.ndarray,
    true_labels: np.ndarray,
    num_thresholds: int = 1000,
) -> float:
    """
    Compute Open-Set Classification Rate (OSCR).

    OSCR measures the ability to correctly classify known samples while
    rejecting unknown samples.

    Reference: "Towards Open Set Deep Networks" (CVPR 2016)

    Args:
        predictions: Predicted class labels [N] (-1 for unknown)
        scores: Confidence/anomaly scores [N]
        true_labels: Ground truth labels [N] (-1 for unknown)
        num_thresholds: Number of thresholds to evaluate

    Returns:
        oscr: Area under OSCR curve
    """
    # Separate known and unknown
    known_mask = true_labels >= 0
    unknown_mask = true_labels < 0

    if known_mask.sum() == 0 or unknown_mask.sum() == 0:
        return 0.0

    known_scores = scores[known_mask]
    unknown_scores = scores[unknown_mask]
    known_preds = predictions[known_mask]
    known_labels = true_labels[known_mask]

    # Generate thresholds
    min_score = min(scores.min(), 0.0)
    max_score = max(scores.max(), 1.0)
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    ccr_list = []  # Correct Classification Rate
    fpr_list = []  # False Positive Rate

    for thresh in thresholds:
        # Classify known samples
        accepted_mask = known_scores >= thresh
        if accepted_mask.sum() > 0:
            ccr = (known_preds[accepted_mask] == known_labels[accepted_mask]).mean()
        else:
            ccr = 0.0

        # FPR on unknown samples
        fpr = (unknown_scores >= thresh).mean()

        ccr_list.append(ccr)
        fpr_list.append(fpr)

    # Compute area under OSCR curve (CCR vs FPR)
    # We want high CCR and low FPR
    oscr = np.trapz(ccr_list, x=fpr_list)

    return abs(oscr)  # Take absolute value


def compute_per_group_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_counts: np.ndarray,
    many_thresh: int = 100,
    few_thresh: int = 20,
) -> Tuple[float, float, float]:
    """
    Compute per-group accuracy for long-tail evaluation.

    Args:
        predictions: Predicted labels [N]
        true_labels: Ground truth labels [N]
        class_counts: Training sample counts per class
        many_thresh: Threshold for many-shot classes
        few_thresh: Threshold for few-shot classes

    Returns:
        many_shot_acc: Accuracy on many-shot classes
        medium_shot_acc: Accuracy on medium-shot classes
        few_shot_acc: Accuracy on few-shot classes
    """
    # Filter to known classes only
    known_mask = true_labels >= 0
    if known_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    predictions = predictions[known_mask]
    true_labels = true_labels[known_mask]

    # Categorize classes
    many_shot_classes = np.where(class_counts >= many_thresh)[0]
    few_shot_classes = np.where(class_counts <= few_thresh)[0]
    medium_shot_classes = np.where((class_counts > few_thresh) & (class_counts < many_thresh))[0]

    # Compute accuracy for each group
    many_shot_acc = 0.0
    medium_shot_acc = 0.0
    few_shot_acc = 0.0

    if len(many_shot_classes) > 0:
        mask = np.isin(true_labels, many_shot_classes)
        if mask.sum() > 0:
            many_shot_acc = (predictions[mask] == true_labels[mask]).mean()

    if len(medium_shot_classes) > 0:
        mask = np.isin(true_labels, medium_shot_classes)
        if mask.sum() > 0:
            medium_shot_acc = (predictions[mask] == true_labels[mask]).mean()

    if len(few_shot_classes) > 0:
        mask = np.isin(true_labels, few_shot_classes)
        if mask.sum() > 0:
            few_shot_acc = (predictions[mask] == true_labels[mask]).mean()

    return many_shot_acc, medium_shot_acc, few_shot_acc


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_openset_recognition(
    predictions: np.ndarray,
    scores: np.ndarray,
    true_labels: np.ndarray,
    class_counts: np.ndarray,
    num_known_classes: int,
    many_thresh: int = 100,
    few_thresh: int = 20,
) -> OpenSetMetrics:
    """
    Comprehensive evaluation for open-set recognition.

    Args:
        predictions: Predicted class labels [N] (-1 for unknown)
        scores: Confidence/anomaly scores [N] (interpretation depends on method)
        true_labels: Ground truth labels [N] (-1 for unknown)
        class_counts: Training sample counts per class
        num_known_classes: Number of known classes
        many_thresh: Threshold for many-shot classes
        few_thresh: Threshold for few-shot classes

    Returns:
        metrics: OpenSetMetrics object with all evaluation results
    """
    # Separate known and unknown
    known_mask = true_labels >= 0
    unknown_mask = true_labels < 0

    num_test_samples = len(true_labels)
    num_unknown_samples = unknown_mask.sum()

    # 1. Closed-set accuracy (on known classes only)
    if known_mask.sum() > 0:
        known_preds = predictions[known_mask]
        known_labels = true_labels[known_mask]
        closed_set_accuracy = (known_preds == known_labels).mean()

        # Per-class accuracy
        per_class_acc = np.zeros(num_known_classes)
        for c in range(num_known_classes):
            mask = known_labels == c
            if mask.sum() > 0:
                per_class_acc[c] = (known_preds[mask] == c).mean()
    else:
        closed_set_accuracy = 0.0
        per_class_acc = np.zeros(num_known_classes)

    # 2. Open-set detection metrics (binary: known vs unknown)
    # Convert to binary labels for detection
    binary_labels = (true_labels < 0).astype(int)  # 1 = unknown, 0 = known

    # Scores interpretation: higher score = more confident in KNOWN class
    # So we need to invert for anomaly detection (higher = more anomalous)
    # This depends on the method, but we assume scores are confidence scores here
    anomaly_scores = -scores  # Negate: now higher = more anomalous

    auroc, aupr = compute_auroc_aupr(anomaly_scores, binary_labels)
    fpr95 = compute_fpr95(anomaly_scores, binary_labels)

    # 3. OSCR (Open-Set Classification Rate)
    oscr = compute_oscr(predictions, scores, true_labels)

    # 4. Overall accuracy (treating unknown as a class)
    overall_accuracy = (predictions == true_labels).mean()

    # 5. Open-set F1 score (binary classification)
    binary_preds = (predictions < 0).astype(int)
    open_set_f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)

    # 6. Long-tail specific metrics
    many_shot_acc, medium_shot_acc, few_shot_acc = compute_per_group_metrics(
        predictions, true_labels, class_counts, many_thresh, few_thresh
    )

    # Package results
    metrics = OpenSetMetrics(
        closed_set_accuracy=closed_set_accuracy,
        closed_set_per_class_acc=per_class_acc,
        auroc=auroc,
        aupr=aupr,
        fpr95=fpr95,
        oscr=oscr,
        open_set_f1=open_set_f1,
        overall_accuracy=overall_accuracy,
        many_shot_acc=many_shot_acc,
        medium_shot_acc=medium_shot_acc,
        few_shot_acc=few_shot_acc,
        num_known_classes=num_known_classes,
        num_test_samples=num_test_samples,
        num_unknown_samples=num_unknown_samples,
    )

    return metrics


def print_metrics(metrics: OpenSetMetrics, title: str = "Open-Set Recognition Results"):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    print(f"\nDataset Info:")
    print(f"  Total test samples: {metrics.num_test_samples}")
    print(f"  Unknown samples: {metrics.num_unknown_samples}")
    print(f"  Known classes: {metrics.num_known_classes}")

    print(f"\nClosed-Set Metrics (Known Classes Only):")
    print(f"  Accuracy: {metrics.closed_set_accuracy:.4f}")

    print(f"\nOpen-Set Detection Metrics:")
    print(f"  AUROC: {metrics.auroc:.4f}")
    print(f"  AUPR: {metrics.aupr:.4f}")
    print(f"  FPR95: {metrics.fpr95:.4f}")

    print(f"\nOpen-Set Classification Metrics:")
    print(f"  OSCR: {metrics.oscr:.4f}")
    print(f"  F1-Score: {metrics.open_set_f1:.4f}")
    print(f"  Overall Accuracy: {metrics.overall_accuracy:.4f}")

    print(f"\nLong-Tail Analysis:")
    print(f"  Many-shot Acc: {metrics.many_shot_acc:.4f}")
    print(f"  Medium-shot Acc: {metrics.medium_shot_acc:.4f}")
    print(f"  Few-shot Acc: {metrics.few_shot_acc:.4f}")

    print("=" * 80 + "\n")


# =============================================================================
# Batch Evaluation Helper
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    openset_detector: any,
    diffusion_model: Optional[torch.nn.Module] = None,
    class_counts: Optional[np.ndarray] = None,
    num_known_classes: Optional[int] = None,
    device: str = "cuda",
    use_diffusion_score: bool = False,
) -> OpenSetMetrics:
    """
    Evaluate model on open-set recognition task.

    Args:
        model: Classification model
        dataloader: Test dataloader
        openset_detector: Open-set detection method (e.g., OpenMax, ODIN, etc.)
        diffusion_model: Feature diffusion model (optional)
        class_counts: Training class counts
        num_known_classes: Number of known classes
        device: Device to use
        use_diffusion_score: Whether to use diffusion reconstruction error as score

    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    if diffusion_model is not None:
        diffusion_model.eval()

    all_predictions = []
    all_scores = []
    all_labels = []

    for batch in dataloader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, *_ = batch

        x = x.to(device)
        y_np = y.cpu().numpy()

        # Forward pass
        if hasattr(model, 'forward_with_features'):
            logits, features = model.forward_with_features(x)
        else:
            # Assume model returns logits
            logits = model(x)
            features = None

        # Get predictions and scores from open-set detector
        if hasattr(openset_detector, 'predict'):
            if use_diffusion_score and diffusion_model is not None and features is not None:
                # Use diffusion reconstruction error as anomaly score
                recon_errors = diffusion_model.diffusion.compute_reconstruction_error(features, y.to(device))
                # Higher error = more anomalous = likely unknown
                # Convert to confidence score (lower error = higher confidence)
                scores_batch = -recon_errors.cpu().numpy()
                # Still need predictions from detector
                _, predictions_batch = openset_detector.predict(logits)
                predictions_batch = predictions_batch.cpu().numpy()
            else:
                scores_batch, predictions_batch = openset_detector.predict(logits, features)
                if isinstance(scores_batch, torch.Tensor):
                    scores_batch = scores_batch.cpu().numpy()
                if isinstance(predictions_batch, torch.Tensor):
                    predictions_batch = predictions_batch.cpu().numpy()
        else:
            # Fallback: use max softmax probability
            probs = torch.softmax(logits, dim=1)
            scores_batch, predictions_batch = torch.max(probs, dim=1)
            scores_batch = scores_batch.cpu().numpy()
            predictions_batch = predictions_batch.cpu().numpy()

        all_predictions.append(predictions_batch)
        all_scores.append(scores_batch)
        all_labels.append(y_np)

    # Concatenate results
    all_predictions = np.concatenate(all_predictions)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Evaluate
    if class_counts is None:
        class_counts = np.ones(num_known_classes)
    if num_known_classes is None:
        num_known_classes = len(class_counts)

    metrics = evaluate_openset_recognition(
        predictions=all_predictions,
        scores=all_scores,
        true_labels=all_labels,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
    )

    return metrics
