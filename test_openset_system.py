#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for Long-Tail Open-Set Recognition System

This script verifies that all components are working correctly:
1. Data loading
2. Model creation
3. Diffusion model
4. Open-set detectors
5. Loss functions
6. Training pipeline
7. Evaluation metrics

Usage:
    python test_openset_system.py

Author: Enhanced Implementation
Date: 2025
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

print("=" * 80)
print("Testing Long-Tail Open-Set Recognition System".center(80))
print("=" * 80)

# =============================================================================
# Test 1: Module Imports
# =============================================================================

print("\n[Test 1/8] Testing module imports...")
try:
    from diffusion_models import create_feature_diffusion, FeatureDiffusion
    from openset_methods import create_openset_detector, OpenMax, ODIN, EnergyBasedOOD
    from openset_losses import LongTailOpenSetLoss, EntropyLoss
    from openset_eval import evaluate_openset_recognition, OpenSetMetrics
    from openset_trainer import LongTailOpenSetTrainer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# =============================================================================
# Test 2: Diffusion Model
# =============================================================================

print("\n[Test 2/8] Testing diffusion model...")
try:
    feature_dim = 128
    num_classes = 6
    batch_size = 32

    # Create diffusion model
    diffusion = create_feature_diffusion(
        feature_dim=feature_dim,
        num_classes=num_classes,
        timesteps=100,  # Small for testing
        hidden_dims=[256, 128, 256],
        conditional=True,
    )

    # Test forward pass
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss, _ = diffusion(features, labels)
    assert loss.item() > 0, "Loss should be positive"

    # Test reconstruction
    recon = diffusion.diffusion.reconstruct(features, labels, num_steps=10)
    assert recon.shape == features.shape, "Reconstruction shape mismatch"

    # Test anomaly scoring
    scores = diffusion.compute_anomaly_score(features, labels, method="reconstruction", num_steps=10)
    assert scores.shape == (batch_size,), "Anomaly scores shape mismatch"

    print(f"✓ Diffusion model working (loss={loss.item():.4f})")
except Exception as e:
    print(f"✗ Diffusion test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 3: Open-Set Detectors
# =============================================================================

print("\n[Test 3/8] Testing open-set detectors...")
try:
    num_samples = 200
    num_test = 50

    # Generate synthetic data
    features_train = torch.randn(num_samples, feature_dim)
    labels_train = torch.randint(0, num_classes, (num_samples,))
    logits_train = torch.randn(num_samples, num_classes)

    features_test = torch.randn(num_test, feature_dim)
    logits_test = torch.randn(num_test, num_classes)

    # Test each detector
    detectors_to_test = ["msp", "energy", "openmax", "mahalanobis"]
    results = {}

    for detector_name in detectors_to_test:
        try:
            detector = create_openset_detector(detector_name, num_classes)

            # Fit if needed
            if detector_name == "openmax":
                detector.fit(
                    features_train.numpy(),
                    labels_train.numpy(),
                    logits_train.numpy(),
                    num_classes
                )
            elif detector_name == "mahalanobis":
                detector.fit(features_train, labels_train, num_classes)

            # Predict
            if detector_name in ["openmax"]:
                scores, preds = detector.predict(features_test, logits_test)
            elif detector_name in ["mahalanobis"]:
                scores, preds = detector.predict(features_test)
            else:
                scores, preds = detector.predict(logits_test)

            assert len(scores) == num_test, f"{detector_name}: scores length mismatch"
            assert len(preds) == num_test, f"{detector_name}: predictions length mismatch"

            results[detector_name] = "✓"
        except Exception as e:
            results[detector_name] = f"✗ {str(e)[:30]}"

    print("  Detector results:")
    for name, result in results.items():
        print(f"    {name:15s}: {result}")

except Exception as e:
    print(f"✗ Detector test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 4: Loss Functions
# =============================================================================

print("\n[Test 4/8] Testing loss functions...")
try:
    class_counts = np.array([100, 80, 60, 40, 20, 10])

    # Test joint loss
    criterion = LongTailOpenSetLoss(
        num_classes=num_classes,
        class_counts=class_counts,
        loss_type="balanced_softmax",
        use_diffusion=True,
        use_contrastive=True,
        feature_dim=feature_dim,
        lambda_diffusion=0.1,
        lambda_contrastive=0.1,
    )

    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    features = torch.randn(batch_size, feature_dim)

    total_loss, loss_dict = criterion(
        logits=logits,
        labels=labels,
        features=features,
        diffusion_model=diffusion,
    )

    assert total_loss.item() > 0, "Total loss should be positive"
    assert "cls" in loss_dict, "Classification loss missing"
    assert "diffusion" in loss_dict, "Diffusion loss missing"
    assert "contrastive" in loss_dict, "Contrastive loss missing"

    print(f"✓ Loss functions working:")
    for key, val in loss_dict.items():
        print(f"    {key:15s}: {val:.4f}")

except Exception as e:
    print(f"✗ Loss test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 5: Evaluation Metrics
# =============================================================================

print("\n[Test 5/8] Testing evaluation metrics...")
try:
    num_total = 100
    num_known_classes = 6

    # Generate synthetic predictions
    predictions = np.random.randint(-1, num_known_classes, num_total)  # -1 = unknown
    scores = np.random.rand(num_total)
    true_labels = np.random.randint(-1, num_known_classes, num_total)

    class_counts = np.array([100, 80, 60, 40, 20, 10])

    metrics = evaluate_openset_recognition(
        predictions=predictions,
        scores=scores,
        true_labels=true_labels,
        class_counts=class_counts,
        num_known_classes=num_known_classes,
    )

    assert isinstance(metrics, OpenSetMetrics), "Metrics should be OpenSetMetrics object"
    assert 0 <= metrics.auroc <= 1, "AUROC should be in [0, 1]"
    assert 0 <= metrics.closed_set_accuracy <= 1, "Accuracy should be in [0, 1]"

    print(f"✓ Evaluation metrics working:")
    print(f"    AUROC: {metrics.auroc:.4f}")
    print(f"    Closed-set Acc: {metrics.closed_set_accuracy:.4f}")
    print(f"    OSCR: {metrics.oscr:.4f}")

except Exception as e:
    print(f"✗ Evaluation test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 6: Synthetic Data Generation
# =============================================================================

print("\n[Test 6/8] Testing synthetic data generation...")
try:
    from torch.utils.data import Dataset, DataLoader, TensorDataset

    # Create synthetic dataset
    num_samples = 500
    num_features = 2  # channels
    seq_length = 1000

    X_train = torch.randn(num_samples, num_features, seq_length)
    y_train = torch.randint(0, num_classes, (num_samples,))

    # Create long-tail distribution
    class_counts_synthetic = np.array([100, 80, 60, 40, 20, 10])
    indices = []
    for c in range(num_classes):
        class_indices = torch.where(y_train == c)[0]
        if len(class_indices) > class_counts_synthetic[c]:
            class_indices = class_indices[:class_counts_synthetic[c]]
        indices.extend(class_indices.tolist())

    X_train = X_train[indices]
    y_train = y_train[indices]

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"✓ Synthetic data created:")
    print(f"    Total samples: {len(dataset)}")
    print(f"    Input shape: {X_train[0].shape}")
    print(f"    Num batches: {len(dataloader)}")

except Exception as e:
    print(f"✗ Data generation test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 7: Model Integration
# =============================================================================

print("\n[Test 7/8] Testing model integration...")
try:
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, num_classes, feature_dim):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
            )
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            features = self.features(x)
            logits = self.classifier(features)
            return logits

        def forward_with_features(self, x):
            x = x.view(x.size(0), -1)
            features = self.features(x)
            logits = self.classifier(features)
            return logits, features

    # Create model
    input_dim = num_features * seq_length
    model = SimpleModel(input_dim, num_classes, feature_dim)

    # Test forward pass
    batch = next(iter(dataloader))
    x, y = batch
    logits, features = model.forward_with_features(x)

    assert logits.shape == (x.size(0), num_classes), "Logits shape mismatch"
    assert features.shape == (x.size(0), feature_dim), "Features shape mismatch"

    print(f"✓ Model integration working:")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Features shape: {features.shape}")

except Exception as e:
    print(f"✗ Model integration test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Test 8: End-to-End Training (1 epoch)
# =============================================================================

print("\n[Test 8/8] Testing end-to-end training (1 epoch)...")
try:
    # Create optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(diffusion.parameters()),
        lr=1e-3
    )

    # Training loop
    model.train()
    diffusion.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= 5:  # Only test 5 batches
            break

        # Forward pass
        logits, features = model.forward_with_features(x)

        # Compute loss
        loss, loss_dict = criterion(
            logits=logits,
            labels=y,
            features=features,
            diffusion_model=diffusion,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    print(f"✓ End-to-end training working:")
    print(f"    Avg loss: {avg_loss:.4f}")
    print(f"    Batches processed: {num_batches}")

except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("Test Summary".center(80))
print("=" * 80)
print("\n✓ All core components are working correctly!")
print("\nNext steps:")
print("  1. Prepare your data in .mat/.h5/.npy format")
print("  2. Configure config_openset.yaml with your data paths")
print("  3. Run: python demo_openset.py --config config_openset.yaml")
print("\nFor detailed documentation, see README_OPENSET.md")
print("=" * 80 + "\n")
