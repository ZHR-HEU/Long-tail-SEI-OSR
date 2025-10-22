#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-Stage Training for Long-Tail Open-Set Recognition

This script implements a two-stage training strategy:

Stage 1: Representation Learning (Closed-Set)
    - Train on known classes only
    - Focus on learning robust features
    - Optional diffusion model for feature enhancement
    - No open-set detection involved

Stage 2: Classifier Retraining + Open-Set Detection
    - Freeze backbone (CRT) or use small LR (fine-tuning)
    - Reinitialize classifier (CRT mode)
    - Apply resampling/reweighting for long-tail
    - Fit and update open-set detector on stable features

Usage:
    python train_openset_twostage.py --config config_openset_twostage.yaml

Author: Two-Stage Open-Set Training Implementation
Date: 2025
"""

import argparse
import os
import random
import sys
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# Import existing modules
from models import ConvNetADSB
from diffusion_models import create_feature_diffusion
from openset_data_utils import create_longtail_openset_dataloaders
from openset_losses import create_longtail_openset_loss
from openset_methods import create_openset_detector
from openset_trainer import LongTailOpenSetTrainer, create_optimizer, create_scheduler
from openset_eval import evaluate_model, print_metrics
from stage2 import (
    find_classifier_layers,
    freeze_backbone_params,
    unfreeze_all_params,
    reinit_classifier_layers,
    set_batchnorm_eval,
    build_stage2_loader,
)


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
    print("Two-Stage Open-Set Training Configuration".center(80))
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


def print_stage_header(stage_num: int, stage_name: str):
    """Print stage header."""
    print("\n" + "=" * 80)
    print(f"STAGE {stage_num}: {stage_name}".center(80))
    print("=" * 80 + "\n")


# =============================================================================
# Model Creation
# =============================================================================

def create_model_with_features(config: Dict[str, Any], num_classes: int, input_shape: tuple) -> nn.Module:
    """
    Create model that returns both logits and features.
    """
    base_model_name = config['model']['name']

    if base_model_name == "ConvNetADSB":
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
# Stage 1: Representation Learning (Closed-Set)
# =============================================================================

def train_stage1(
    config: Dict[str, Any],
    train_loader,
    val_loader,
    data_info: Dict[str, Any],
    device: str,
) -> tuple[nn.Module, Optional[nn.Module], int]:
    """
    Stage 1: Representation Learning on Known Classes Only.

    Returns:
        model: Trained model
        diffusion_model: Trained diffusion model (if enabled)
        feature_dim: Feature dimension
    """
    print_stage_header(1, "Representation Learning (Closed-Set)")

    num_known_classes = data_info['num_known_classes']
    class_counts = data_info['class_counts']

    # Get input shape
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape

    # Create model
    print("Creating Model...")
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

    # Create diffusion model (if enabled)
    diffusion_model = None
    if config['stage1']['diffusion']['enabled']:
        print("\nCreating Diffusion Model for Stage-1...")
        diffusion_config = config['stage1']['diffusion']
        diffusion_model = create_feature_diffusion(
            feature_dim=feature_dim,
            num_classes=num_known_classes,
            hidden_dims=diffusion_config['hidden_dims'],
            timesteps=diffusion_config['timesteps'],
            beta_schedule=diffusion_config['beta_schedule'],
            conditional=diffusion_config['conditional'],
            dropout=diffusion_config['dropout'],
            enhanced=diffusion_config['enhanced'],
        )
        diffusion_model = diffusion_model.to(device)
        print("Diffusion model created")

    # Create loss function (closed-set only)
    print("\nCreating Loss Function (Closed-Set)...")
    criterion = create_longtail_openset_loss(
        num_classes=num_known_classes,
        class_counts=class_counts,
        loss_config=config['stage1']['loss'],
    )

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        diffusion_model=diffusion_model,
        lr=config['stage1']['lr'],
        weight_decay=config['stage1']['weight_decay'],
        optimizer_type=config['stage1']['optimizer'],
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_epochs=config['stage1']['epochs'],
        scheduler_type=config['stage1']['scheduler'],
        warmup_epochs=config['stage1']['warmup_epochs'],
    )

    # Create trainer
    print("\nCreating Trainer for Stage-1...")
    os.makedirs(config['stage1']['checkpoint_dir'], exist_ok=True)
    trainer = LongTailOpenSetTrainer(
        model=model,
        diffusion_model=diffusion_model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_interval=config['stage1']['console_log_interval'],
        checkpoint_dir=config['stage1']['checkpoint_dir'],
    )

    # Train Stage-1 (WITHOUT open-set detector)
    print("\n" + "-" * 80)
    print("Training Stage-1 (Closed-Set Classification Only)")
    print("-" * 80)

    # Use a dummy detector (not actually used in Stage-1)
    dummy_detector = create_openset_detector(
        method="msp",
        num_classes=num_known_classes,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['stage1']['epochs'],
        openset_detector=dummy_detector,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
        scheduler=scheduler,
        early_stopping_patience=config['stage1']['early_stopping_patience'],
        metric_for_best="closed_set_acc",  # Use closed-set accuracy for Stage-1
    )

    # Load best model from Stage-1
    print("\n" + "-" * 80)
    print("Loading Best Stage-1 Model")
    print("-" * 80)
    trainer.load_checkpoint("best_model.pth")

    print("\n[Stage-1 Completed Successfully]")
    print(f"Best model saved in: {config['stage1']['checkpoint_dir']}")

    return model, diffusion_model, feature_dim


# =============================================================================
# Stage 2: Classifier Retraining + Open-Set Detection
# =============================================================================

def train_stage2(
    config: Dict[str, Any],
    model: nn.Module,
    diffusion_model: Optional[nn.Module],
    feature_dim: int,
    train_dataset,
    val_loader,
    test_loader,
    data_info: Dict[str, Any],
    device: str,
):
    """
    Stage 2: Classifier Retraining + Open-Set Detection.

    Args:
        model: Pre-trained model from Stage-1
        diffusion_model: Pre-trained diffusion model from Stage-1 (if any)
        feature_dim: Feature dimension
        train_dataset: Training dataset (for creating new loader with resampling)
        val_loader: Validation loader
        test_loader: Test loader
        data_info: Dataset information
        device: Device
    """
    print_stage_header(2, "Classifier Retraining + Open-Set Detection")

    num_known_classes = data_info['num_known_classes']
    class_counts = data_info['class_counts']

    # -------------------------------------------------------------------------
    # Step 1: Prepare Model for Stage-2
    # -------------------------------------------------------------------------
    print("Preparing model for Stage-2...")
    print(f"Mode: {config['stage2']['mode']}")

    # Find classifier layers
    classifier_layers_info = find_classifier_layers(model, num_known_classes)
    if not classifier_layers_info:
        raise ValueError("Could not find classifier layers in the model")

    classifier_names = [name for name, _ in classifier_layers_info]
    classifier_layers = [layer for _, layer in classifier_layers_info]

    print(f"Classifier layers found: {classifier_names}")

    # Apply stage-2 mode
    if config['stage2']['mode'] == "crt":
        print("\n[CRT Mode]")
        print("  - Freezing backbone parameters")
        print("  - Reinitializing classifier layers")

        freeze_backbone_params(model, classifier_names)
        reinit_classifier_layers(classifier_layers)

        # Freeze BatchNorm if configured
        if config['stage2']['bn_mode'] == "freeze":
            print("  - Freezing BatchNorm statistics")
            model.apply(set_batchnorm_eval)

    elif config['stage2']['mode'] == "finetune":
        print("\n[Fine-tuning Mode]")
        print("  - Unfreezing all parameters")
        print("  - Using different LRs for backbone vs classifier")

        unfreeze_all_params(model)

    else:
        raise ValueError(f"Unknown stage2 mode: {config['stage2']['mode']}")

    # -------------------------------------------------------------------------
    # Step 2: Create Stage-2 DataLoader with Resampling
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Creating Stage-2 DataLoader with Resampling")
    print("-" * 80)

    sampler_config = {
        'name': config['stage2']['sampling_strategy'],
        'alpha_start': config['stage2'].get('alpha_start', 0.5),
        'alpha_end': config['stage2'].get('alpha_end', 0.0),
    }

    train_loader_stage2 = build_stage2_loader(
        dataset=train_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        sampler_config=sampler_config,
        total_epochs=config['stage2']['epochs'],
        seed=config['seed'],
    )

    print(f"Sampling strategy: {config['stage2']['sampling_strategy']}")
    if config['stage2']['sampling_strategy'] == 'progressive_power':
        print(f"  Alpha range: {config['stage2']['alpha_start']} -> {config['stage2']['alpha_end']}")

    # -------------------------------------------------------------------------
    # Step 3: Create Loss Function for Stage-2
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Creating Stage-2 Loss Function")
    print("-" * 80)

    criterion_stage2 = create_longtail_openset_loss(
        num_classes=num_known_classes,
        class_counts=class_counts,
        loss_config=config['stage2']['loss'],
    )

    # -------------------------------------------------------------------------
    # Step 4: Create Optimizer for Stage-2
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Creating Stage-2 Optimizer")
    print("-" * 80)

    if config['stage2']['mode'] == "crt":
        # Only optimize classifier parameters
        params_to_optimize = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)

        print(f"Optimizing {len(params_to_optimize)} parameter groups (classifier only)")

        if config['stage2']['optimizer'] == "Adam":
            optimizer_stage2 = torch.optim.Adam(
                params_to_optimize,
                lr=config['stage2']['lr_classifier'],
                weight_decay=config['stage2']['weight_decay'],
            )
        elif config['stage2']['optimizer'] == "SGD":
            optimizer_stage2 = torch.optim.SGD(
                params_to_optimize,
                lr=config['stage2']['lr_classifier'],
                weight_decay=config['stage2']['weight_decay'],
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['stage2']['optimizer']}")

    elif config['stage2']['mode'] == "finetune":
        # Use different LRs for backbone and classifier
        backbone_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if any(name.startswith(cn) for cn in classifier_names):
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        print(f"Backbone parameters: {len(backbone_params)}")
        print(f"Classifier parameters: {len(classifier_params)}")
        print(f"  Backbone LR: {config['stage2']['lr_backbone']}")
        print(f"  Classifier LR: {config['stage2']['lr_classifier']}")

        if config['stage2']['optimizer'] == "Adam":
            optimizer_stage2 = torch.optim.Adam([
                {'params': backbone_params, 'lr': config['stage2']['lr_backbone']},
                {'params': classifier_params, 'lr': config['stage2']['lr_classifier']},
            ], weight_decay=config['stage2']['weight_decay'])
        elif config['stage2']['optimizer'] == "SGD":
            optimizer_stage2 = torch.optim.SGD([
                {'params': backbone_params, 'lr': config['stage2']['lr_backbone']},
                {'params': classifier_params, 'lr': config['stage2']['lr_classifier']},
            ], weight_decay=config['stage2']['weight_decay'], momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config['stage2']['optimizer']}")

    # -------------------------------------------------------------------------
    # Step 5: Create Scheduler for Stage-2
    # -------------------------------------------------------------------------
    scheduler_stage2 = create_scheduler(
        optimizer=optimizer_stage2,
        num_epochs=config['stage2']['epochs'],
        scheduler_type=config['stage2']['scheduler'],
        warmup_epochs=config['stage2']['warmup_epochs'],
    )

    # -------------------------------------------------------------------------
    # Step 6: Create Diffusion Model for Stage-2 (if enabled)
    # -------------------------------------------------------------------------
    diffusion_model_stage2 = None
    if config['stage2']['diffusion']['enabled']:
        print("\n" + "-" * 80)
        print("Using Diffusion Model in Stage-2")
        print("-" * 80)
        # Reuse the diffusion model from Stage-1 if available
        if diffusion_model is not None:
            diffusion_model_stage2 = diffusion_model
            print("Reusing diffusion model from Stage-1")
        else:
            # Create new diffusion model
            diffusion_config = config['stage2']['diffusion']
            diffusion_model_stage2 = create_feature_diffusion(
                feature_dim=feature_dim,
                num_classes=num_known_classes,
                hidden_dims=config['stage1']['diffusion']['hidden_dims'],
                timesteps=config['stage1']['diffusion']['timesteps'],
                beta_schedule=config['stage1']['diffusion']['beta_schedule'],
                conditional=config['stage1']['diffusion']['conditional'],
                dropout=config['stage1']['diffusion']['dropout'],
                enhanced=config['stage1']['diffusion']['enhanced'],
            )
            diffusion_model_stage2 = diffusion_model_stage2.to(device)
            print("Created new diffusion model for Stage-2")

    # -------------------------------------------------------------------------
    # Step 7: Create Trainer for Stage-2
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Creating Trainer for Stage-2")
    print("-" * 80)

    os.makedirs(config['stage2']['checkpoint_dir'], exist_ok=True)
    trainer_stage2 = LongTailOpenSetTrainer(
        model=model,
        diffusion_model=diffusion_model_stage2,
        criterion=criterion_stage2,
        optimizer=optimizer_stage2,
        device=device,
        log_interval=config['stage2']['console_log_interval'],
        checkpoint_dir=config['stage2']['checkpoint_dir'],
    )

    # -------------------------------------------------------------------------
    # Step 8: Fit Open-Set Detector (Initial Fit on Stage-1 Features)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Fitting Open-Set Detector on Stage-1 Features")
    print("-" * 80)

    detector_type = config['stage2']['openset']['detector_type']
    detector_config = config['stage2']['openset'].get(detector_type, {})

    print(f"Detector type: {detector_type}")

    openset_detector = trainer_stage2.fit_openset_detector(
        train_loader=train_loader_stage2,
        detector_type=detector_type,
        **detector_config,
    )

    # -------------------------------------------------------------------------
    # Step 9: Train Stage-2
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Training Stage-2 (Classifier Retraining + Open-Set Detection)")
    print("-" * 80)

    # Modify the trainer to handle detector refitting
    original_train = trainer_stage2.train

    def train_with_detector_refitting(*args, **kwargs):
        """Wrapper to handle periodic detector refitting."""
        refit_interval = config['stage2']['openset'].get('refit_interval', 20)

        # Get the original train function parameters
        train_loader = kwargs.get('train_loader') or args[0]
        val_loader = kwargs.get('val_loader') or args[1]
        num_epochs = kwargs.get('num_epochs') or args[2]
        current_detector = kwargs.get('openset_detector') or args[3]
        class_counts = kwargs.get('class_counts') or args[4]
        num_known_classes = kwargs.get('num_known_classes') or args[5]

        # Replace the detector refitting logic in the train loop
        # We'll do this by modifying the history after each epoch
        # This is a simplified version - in production you'd want to modify the train loop directly
        return original_train(*args, **kwargs)

    # Train Stage-2
    openset_detector = trainer_stage2.train(
        train_loader=train_loader_stage2,
        val_loader=val_loader,
        num_epochs=config['stage2']['epochs'],
        openset_detector=openset_detector,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
        scheduler=scheduler_stage2,
        early_stopping_patience=config['stage2']['early_stopping_patience'],
        metric_for_best=config['stage2']['metric_for_best'],
    )

    # -------------------------------------------------------------------------
    # Step 10: Load Best Stage-2 Model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Loading Best Stage-2 Model")
    print("-" * 80)
    trainer_stage2.load_checkpoint("best_model.pth")

    # Re-fit detector on best model
    print("\n" + "-" * 80)
    print("Re-fitting Detector on Best Stage-2 Model")
    print("-" * 80)
    openset_detector = trainer_stage2.fit_openset_detector(
        train_loader=train_loader_stage2,
        detector_type=detector_type,
        **detector_config,
    )

    # -------------------------------------------------------------------------
    # Step 11: Final Evaluation on Test Set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        openset_detector=openset_detector,
        diffusion_model=diffusion_model_stage2,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
        device=device,
        use_diffusion_score=config['evaluation'].get('use_diffusion_score', False),
    )

    print_metrics(test_metrics, title="Test Set Results (After Two-Stage Training)")

    # Save results
    results_dir = config['stage2']['checkpoint_dir']
    results_file = os.path.join(results_dir, "final_results.txt")

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Two-Stage Open-Set Training - Final Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment: {config['exp_name']}\n")
        f.write(f"Stage-2 Mode: {config['stage2']['mode']}\n")
        f.write(f"Detector Type: {detector_type}\n\n")
        f.write("Closed-Set Performance:\n")
        f.write(f"  Closed-Set Accuracy: {test_metrics.closed_set_accuracy:.4f}\n")
        f.write(f"  Overall Accuracy: {test_metrics.overall_accuracy:.4f}\n\n")
        f.write("Open-Set Performance:\n")
        f.write(f"  AUROC: {test_metrics.auroc:.4f}\n")
        f.write(f"  AUPR: {test_metrics.aupr:.4f}\n")
        f.write(f"  FPR95: {test_metrics.fpr95:.4f}\n")
        f.write(f"  OSCR: {test_metrics.oscr:.4f}\n")
        f.write(f"  F1-Score: {test_metrics.open_set_f1:.4f}\n\n")
        f.write("Long-Tail Analysis:\n")
        f.write(f"  Many-shot Acc: {test_metrics.many_shot_acc:.4f}\n")
        f.write(f"  Medium-shot Acc: {test_metrics.medium_shot_acc:.4f}\n")
        f.write(f"  Few-shot Acc: {test_metrics.few_shot_acc:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    print("\n[Stage-2 Completed Successfully]")


# =============================================================================
# Main Pipeline
# =============================================================================

def main(config: Dict[str, Any]):
    """Main two-stage training pipeline."""

    print("\n" + "=" * 80)
    print("Two-Stage Long-Tail Open-Set Recognition".center(80))
    print("=" * 80)

    # Set seed
    set_seed(config['seed'])

    # Device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"\nDevice: {device}")

    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
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
        augmentation=config['stage1']['augmentation'],  # Use Stage-1 augmentation for initial loaders
        sampling_strategy=config['stage1']['sampling_strategy'],
        seed=config['seed'],
    )

    # Get the training dataset (we'll need it to create Stage-2 loader)
    train_dataset = train_loader.dataset

    print(f"\nDataset Info:")
    print(f"  Known classes: {data_info['num_known_classes']}")
    print(f"  Unknown classes: {data_info['num_unknown_classes']}")
    print(f"  Imbalance ratio: {data_info['imbalance_ratio']:.2f}")
    print(f"  Class counts: {data_info['class_counts']}")

    # -------------------------------------------------------------------------
    # Stage 1: Representation Learning
    # -------------------------------------------------------------------------
    if config['stage1']['enabled']:
        model, diffusion_model, feature_dim = train_stage1(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            data_info=data_info,
            device=device,
        )
    else:
        raise ValueError("Stage-1 must be enabled for two-stage training")

    # -------------------------------------------------------------------------
    # Stage 2: Classifier Retraining + Open-Set Detection
    # -------------------------------------------------------------------------
    if config['stage2']['enabled']:
        train_stage2(
            config=config,
            model=model,
            diffusion_model=diffusion_model,
            feature_dim=feature_dim,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            data_info=data_info,
            device=device,
        )
    else:
        print("\n[Stage-2 Disabled - Skipping]")

    print("\n" + "=" * 80)
    print("Two-Stage Training Pipeline Completed Successfully!")
    print("=" * 80 + "\n")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Stage Training for Long-Tail Open-Set Recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_openset_twostage.yaml",
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
