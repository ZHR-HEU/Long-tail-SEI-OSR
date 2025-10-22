# -*- coding: utf-8 -*-
"""
Training Pipeline for Long-Tail Open-Set Recognition

This module implements the complete training pipeline for long-tail open-set recognition:
1. Joint training of classifier + diffusion model
2. Feature extraction and open-set detector fitting
3. Multi-stage training (baseline -> fine-tuning -> calibration)
4. Comprehensive logging and checkpointing

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import components
from openset_losses import LongTailOpenSetLoss
from openset_eval import evaluate_model, print_metrics
from openset_methods import create_openset_detector


# =============================================================================
# Trainer Class
# =============================================================================

class LongTailOpenSetTrainer:
    """
    Trainer for long-tail open-set recognition.

    Handles:
    - Joint training of classification model and diffusion model
    - Feature extraction for open-set detector fitting
    - Multi-stage training with different objectives
    - Logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion_model: Optional[nn.Module],
        criterion: LongTailOpenSetLoss,
        optimizer: optim.Optimizer,
        device: str = "cuda",
        log_interval: int = 10,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Args:
            model: Classification model
            diffusion_model: Feature diffusion model (optional)
            criterion: Joint loss function
            optimizer: Optimizer
            device: Device to use
            log_interval: Logging interval (epochs)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.diffusion_model = diffusion_model.to(device) if diffusion_model is not None else None
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.history = {
            "train_loss": [],
            "val_metrics": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.diffusion_model is not None:
            self.diffusion_model.train()

        total_loss = 0.0
        loss_components = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            x, y = batch[:2]
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            if hasattr(self.model, 'forward_with_features'):
                logits, features = self.model.forward_with_features(x)
            else:
                logits = self.model(x)
                features = None

            # Compute loss
            loss, loss_dict = self.criterion(
                logits=logits,
                labels=y,
                features=features,
                diffusion_model=self.diffusion_model,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val

            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}

        return {"loss": avg_loss, **avg_components}

    @torch.no_grad()
    def extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from data loader.

        Returns:
            features: Feature vectors [N, D]
            logits: Model logits [N, C]
            labels: Ground truth labels [N]
        """
        self.model.eval()

        all_features = []
        all_logits = []
        all_labels = []

        for batch in tqdm(loader, desc="Extracting features"):
            x, y = batch[:2]
            x = x.to(self.device)

            if hasattr(self.model, 'forward_with_features'):
                logits, features = self.model.forward_with_features(x)
            else:
                raise ValueError("Model must implement forward_with_features() for feature extraction")

            all_features.append(features.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, logits, labels

    def fit_openset_detector(
        self,
        train_loader: DataLoader,
        detector_type: str = "openmax",
        **detector_kwargs,
    ) -> Any:
        """
        Fit open-set detector using training features.

        Args:
            train_loader: Training data loader
            detector_type: Type of detector ("openmax", "mahalanobis", etc.)
            **detector_kwargs: Detector-specific arguments

        Returns:
            Fitted detector
        """
        print(f"\n[Fitting {detector_type} detector]")

        # Extract features
        features, logits, labels = self.extract_features(train_loader)

        # Create and fit detector
        num_classes = logits.shape[1]
        detector = create_openset_detector(
            method=detector_type,
            num_classes=num_classes,
            **detector_kwargs,
        )

        if detector_type == "openmax":
            detector.fit(features, labels, logits, num_classes)
        elif detector_type == "mahalanobis":
            features_torch = torch.from_numpy(features).to(self.device)
            labels_torch = torch.from_numpy(labels).to(self.device)
            detector.fit(features_torch, labels_torch, num_classes)
        # Other detectors don't need fitting

        print(f"[Detector fitted successfully]")
        return detector

    def validate(
        self,
        val_loader: DataLoader,
        openset_detector: Any,
        class_counts: np.ndarray,
        num_known_classes: int,
    ) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            val_loader: Validation data loader
            openset_detector: Fitted open-set detector
            class_counts: Training class counts
            num_known_classes: Number of known classes

        Returns:
            Dictionary of validation metrics
        """
        metrics = evaluate_model(
            model=self.model,
            dataloader=val_loader,
            openset_detector=openset_detector,
            diffusion_model=self.diffusion_model,
            class_counts=class_counts,
            num_known_classes=num_known_classes,
            device=self.device,
            use_diffusion_score=False,
        )

        return {
            "closed_set_acc": metrics.closed_set_accuracy,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "oscr": metrics.oscr,
            "overall_acc": metrics.overall_accuracy,
            "many_shot_acc": metrics.many_shot_acc,
            "few_shot_acc": metrics.few_shot_acc,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        openset_detector: Any,
        class_counts: np.ndarray,
        num_known_classes: int,
        scheduler: Optional[Any] = None,
        early_stopping_patience: int = 20,
        metric_for_best: str = "oscr",
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            openset_detector: Open-set detector (will be re-fitted periodically)
            class_counts: Training class counts
            num_known_classes: Number of known classes
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping
            metric_for_best: Metric to use for selecting best model
        """
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)

        best_metric = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])

            # Validate
            if (epoch + 1) % self.log_interval == 0 or epoch == num_epochs - 1:
                print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")

                # Re-fit detector periodically
                if epoch % 20 == 0 and epoch > 0:
                    openset_detector = self.fit_openset_detector(train_loader, detector_type="openmax")

                val_metrics = self.validate(
                    val_loader, openset_detector, class_counts, num_known_classes
                )
                self.history["val_metrics"].append(val_metrics)

                print(f"  Val Metrics:")
                print(f"    Closed-Set Acc: {val_metrics['closed_set_acc']:.4f}")
                print(f"    AUROC: {val_metrics['auroc']:.4f}")
                print(f"    OSCR: {val_metrics['oscr']:.4f}")
                print(f"    Overall Acc: {val_metrics['overall_acc']:.4f}")

                # Check for improvement
                current_metric = val_metrics[metric_for_best]
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint("best_model.pth", val_metrics)
                    print(f"  -> New best {metric_for_best}: {current_metric:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Early stopping triggered after {epoch + 1} epochs]")
                    break

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()

            # Save periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        print("\n" + "=" * 80)
        print("Training Completed")
        print("=" * 80)
        print(f"Best {metric_for_best}: {best_metric:.4f}")

        return openset_detector

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }

        if self.diffusion_model is not None:
            checkpoint["diffusion_state_dict"] = self.diffusion_model.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.history = checkpoint.get("history", self.history)

        if self.diffusion_model is not None and "diffusion_state_dict" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])

        print(f"Checkpoint loaded: {path}")


# =============================================================================
# Helper Functions
# =============================================================================

def create_optimizer(
    model: nn.Module,
    diffusion_model: Optional[nn.Module],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_type: str = "Adam",
) -> optim.Optimizer:
    """Create optimizer for model and diffusion model."""
    params = list(model.parameters())

    if diffusion_model is not None:
        params += list(diffusion_model.parameters())

    if optimizer_type == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 5,
) -> Optional[Any]:
    """Create learning rate scheduler."""
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return scheduler
