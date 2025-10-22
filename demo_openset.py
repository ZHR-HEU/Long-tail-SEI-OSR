#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Demo Script for Long-Tail Open-Set Recognition with Diffusion Models

This script demonstrates the full pipeline:
1. Load data with long-tail distribution
2. Split into known/unknown classes
3. Train classifier with diffusion model
4. Fit open-set detector
5. Evaluate on test set
6. Visualize results

Usage:
    python demo_openset.py --config config_openset.yaml

Author: Enhanced Implementation
Date: 2025
"""

import argparse
import os
import random
import sys
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import yaml

# Import modules
from models import ConvNetADSB
from diffusion_models import create_feature_diffusion
from openset_data_utils import create_longtail_openset_dataloaders
from openset_losses import create_longtail_openset_loss
from openset_methods import create_openset_detector
from openset_trainer import LongTailOpenSetTrainer, create_optimizer, create_scheduler
from openset_eval import evaluate_model, print_metrics


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_config(config: Dict[str, Any]):
    """Print configuration."""
    print("\n" + "=" * 80)
    print("Configuration".center(80))
    print("=" * 80)

    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_dict(config)
    print("=" * 80 + "\n")


# =============================================================================
# Model Creation
# =============================================================================

def create_model_with_features(config: Dict[str, Any], num_classes: int, input_shape: tuple) -> nn.Module:
    """
    Create model that returns both logits and features.

    We'll wrap the existing model to expose features.
    """
    base_model_name = config['model']['name']

    if base_model_name == "ConvNetADSB":
        # ``ConvNetADSB`` expects the keyword ``dropout_rate``.  The demo
        # configuration, however, follows the convention used elsewhere in the
        # project (`dropout`).  To stay backward compatible we transparently
        # map both keys here.
        dropout_rate = config['model'].get(
            'dropout_rate',
            config['model'].get('dropout', 0.0),
        )

        model = ConvNetADSB(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_attention=config['model']['use_attention'],
            norm_kind=config['model']['norm_kind'],
        )
    else:
        raise ValueError(f"Unknown model: {base_model_name}")

    # Wrap to expose features
    class ModelWithFeatures(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            logits, _ = self.base_model(x)
            return logits

        def forward_with_features(self, x):
            """Forward pass returning both logits and features."""
            logits, features = self.base_model(x)
            return logits, features

    wrapped_model = ModelWithFeatures(model)
    return wrapped_model


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main(config: Dict[str, Any]):
    """Main training pipeline."""

    print("\n" + "=" * 80)
    print("Long-Tail Open-Set Recognition with Diffusion Models".center(80))
    print("=" * 80)

    # Set seed
    set_seed(config['seed'])

    # Device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\nDevice: {device}")

    # Create dataloaders
    print("\n" + "-" * 80)
    print("Creating Dataloaders")
    print("-" * 80)

    train_loader, val_loader, test_loader, data_info = create_longtail_openset_dataloaders(
        data_path=config['data']['path_train'],
        num_known_classes=config['data']['num_known_classes'],
        split_protocol=config['data']['split_protocol'],
        imbalance_ratio=config['data']['imbalance_ratio'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        target_length=config['data']['target_length'],
        normalize=config['data']['normalize'],
        augmentation=config['data']['augmentation'],
        sampling_strategy=config['data']['sampling_strategy'],
        seed=config['seed'],
        alpha_start=config['data'].get('alpha_start', 0.5),
        alpha_end=config['data'].get('alpha_end', 0.0),
    )

    num_known_classes = data_info['num_known_classes']
    class_counts = data_info['class_counts']

    print(f"\nDataset Info:")
    print(f"  Known classes: {num_known_classes}")
    print(f"  Unknown classes: {data_info['num_unknown_classes']}")
    print(f"  Imbalance ratio: {data_info['imbalance_ratio']:.2f}")
    print(f"  Class counts: {class_counts}")

    # Get input shape from first batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape  # [B, C, T]
    print(f"  Input shape: {input_shape}")

    # Create model
    print("\n" + "-" * 80)
    print("Creating Model")
    print("-" * 80)

    model = create_model_with_features(
        config=config,
        num_classes=num_known_classes,
        input_shape=input_shape[1:],
    )
    model = model.to(device)

    print(f"Model created: {config['model']['name']}")

    # Get feature dimension
    with torch.no_grad():
        sample_input = sample_batch[0][:2].to(device)
        _, sample_features = model.forward_with_features(sample_input)
        feature_dim = sample_features.shape[1]
        print(f"Feature dimension: {feature_dim}")

    # Create diffusion model
    diffusion_model = None
    if config['diffusion']['enabled']:
        print("\n" + "-" * 80)
        print("Creating Diffusion Model")
        print("-" * 80)

        diffusion_model = create_feature_diffusion(
            feature_dim=feature_dim,
            num_classes=num_known_classes,
            hidden_dims=config['diffusion']['hidden_dims'],
            timesteps=config['diffusion']['timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            conditional=config['diffusion']['conditional'],
            dropout=config['diffusion']['dropout'],
            enhanced=config['diffusion']['enhanced'],
        )

    # Create loss function
    print("\n" + "-" * 80)
    print("Creating Loss Function")
    print("-" * 80)

    criterion = create_longtail_openset_loss(
        num_classes=num_known_classes,
        class_counts=class_counts,
        loss_config=config['loss'],
    )

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        diffusion_model=diffusion_model,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        optimizer_type=config['training']['optimizer'],
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_epochs=config['training']['epochs'],
        scheduler_type=config['training']['scheduler'],
        warmup_epochs=config['training']['warmup_epochs'],
    )

    # Create trainer
    print("\n" + "-" * 80)
    print("Creating Trainer")
    print("-" * 80)

    trainer = LongTailOpenSetTrainer(
        model=model,
        diffusion_model=diffusion_model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_interval=config['console_log_interval'],
        checkpoint_dir=config['training']['checkpoint_dir'],
    )

    # Create initial open-set detector
    detector_type = config['openset']['detector_type']
    detector_config = config['openset'].get(detector_type, {})

    print(f"\nOpen-set detector: {detector_type}")

    # Fit detector on initial features
    openset_detector = trainer.fit_openset_detector(
        train_loader=train_loader,
        detector_type=detector_type,
        **detector_config,
    )

    # Train
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    openset_detector = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        openset_detector=openset_detector,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
        scheduler=scheduler,
        early_stopping_patience=config['training']['early_stopping_patience'],
        metric_for_best=config['training']['metric_for_best'],
    )

    # Load best model
    print("\n" + "-" * 80)
    print("Loading Best Model")
    print("-" * 80)

    trainer.load_checkpoint("best_model.pth")

    # Re-fit detector on best model
    openset_detector = trainer.fit_openset_detector(
        train_loader=train_loader,
        detector_type=detector_type,
        **detector_config,
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        openset_detector=openset_detector,
        diffusion_model=diffusion_model,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
        device=device,
        use_diffusion_score=config['diffusion']['enabled'],
    )

    print_metrics(test_metrics, title="Test Set Results")

    # Save results
    results_dir = config['training']['checkpoint_dir']
    results_file = os.path.join(results_dir, "final_results.txt")

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Final Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Closed-Set Accuracy: {test_metrics.closed_set_accuracy:.4f}\n")
        f.write(f"AUROC: {test_metrics.auroc:.4f}\n")
        f.write(f"AUPR: {test_metrics.aupr:.4f}\n")
        f.write(f"FPR95: {test_metrics.fpr95:.4f}\n")
        f.write(f"OSCR: {test_metrics.oscr:.4f}\n")
        f.write(f"F1-Score: {test_metrics.open_set_f1:.4f}\n")
        f.write(f"Overall Accuracy: {test_metrics.overall_accuracy:.4f}\n")
        f.write(f"\nLong-Tail Analysis:\n")
        f.write(f"  Many-shot Acc: {test_metrics.many_shot_acc:.4f}\n")
        f.write(f"  Medium-shot Acc: {test_metrics.medium_shot_acc:.4f}\n")
        f.write(f"  Few-shot Acc: {test_metrics.few_shot_acc:.4f}\n")

    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-Tail Open-Set Recognition Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config_openset.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    print_config(config)

    # Run main pipeline
    try:
        main(config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
