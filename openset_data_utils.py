# -*- coding: utf-8 -*-
"""
Open-Set Data Utilities for Long-Tail Learning

This module extends the base data_utils.py to support open-set recognition scenarios:
1. Split dataset into known/unknown classes
2. Create long-tail distribution for known classes
3. Provide unknown class samples for evaluation
4. Support multiple open-set protocols (standard, cross-dataset, etc.)

Open-Set Protocols:
- Standard: Random split of classes into known/unknown
- Cross-Dataset: Use different dataset as unknown
- Hard: Select confusing classes as unknown (e.g., similar modulations)

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Import base utilities
from data_utils import (
    SignalDataset,
    get_sampler,
    Compose,
    ToTensor,
    RandomTimeShift,
    RandomAmplitude,
    RandomNoise,
    PerSampleNormalize,
)


# =============================================================================
# Open-Set Dataset Wrapper
# =============================================================================

class OpenSetDataset(Dataset):
    """
    Wrapper for open-set scenarios: combines known and unknown class samples.

    Args:
        known_dataset: Dataset containing known classes
        unknown_dataset: Dataset containing unknown classes (optional, for evaluation)
        known_label_map: Mapping from original labels to new labels [0, num_known-1]
        return_original_label: Whether to return original label (for analysis)
    """

    def __init__(
        self,
        known_dataset: Dataset,
        unknown_dataset: Optional[Dataset] = None,
        known_label_map: Optional[Dict[int, int]] = None,
        return_original_label: bool = False,
    ):
        self.known_dataset = known_dataset
        self.unknown_dataset = unknown_dataset
        self.known_label_map = known_label_map or {}
        self.return_original_label = return_original_label

        # Compute dataset info
        self.num_known = len(known_dataset)
        self.num_unknown = len(unknown_dataset) if unknown_dataset is not None else 0
        self.total_len = self.num_known + self.num_unknown

        # Get number of known classes
        if hasattr(known_dataset, 'labels'):
            unique_labels = np.unique(known_dataset.labels)
            self.num_known_classes = len(unique_labels)
        else:
            self.num_known_classes = len(set(self.known_label_map.values())) if known_label_map else None

        print(f"[OpenSetDataset] Created dataset:")
        print(f"  - Known samples: {self.num_known}")
        print(f"  - Unknown samples: {self.num_unknown}")
        print(f"  - Known classes: {self.num_known_classes}")

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, int]:
        if idx < self.num_known:
            # Known class sample
            x, y = self.known_dataset[idx]
            original_y = y

            # Remap label
            if self.known_label_map:
                y = self.known_label_map.get(int(y), int(y))

            is_known = 1
        else:
            # Unknown class sample
            if self.unknown_dataset is None:
                raise IndexError(f"Index {idx} out of range (no unknown dataset)")

            unknown_idx = idx - self.num_known
            x, y = self.unknown_dataset[unknown_idx]
            original_y = y

            # Label unknown as -1
            y = -1
            is_known = 0

        if self.return_original_label:
            return x, y, original_y, is_known
        else:
            return x, y

    @property
    def labels(self) -> np.ndarray:
        """Return labels (for compatibility with sampler)."""
        if hasattr(self.known_dataset, 'labels'):
            known_labels = self.known_dataset.labels
            # Remap if necessary
            if self.known_label_map:
                known_labels = np.array([self.known_label_map.get(int(l), int(l)) for l in known_labels])
        else:
            known_labels = np.array([self.known_dataset[i][1] for i in range(len(self.known_dataset))])

        if self.unknown_dataset is not None:
            unknown_labels = np.full(len(self.unknown_dataset), -1, dtype=known_labels.dtype)
            return np.concatenate([known_labels, unknown_labels])
        else:
            return known_labels

    @property
    def num_classes(self) -> int:
        """Return number of known classes."""
        return self.num_known_classes

    @property
    def class_counts(self) -> np.ndarray:
        """Return class counts (only for known classes)."""
        labels = self.labels
        known_mask = labels >= 0
        known_labels = labels[known_mask]

        if len(known_labels) == 0:
            return np.array([])

        counts = np.bincount(known_labels, minlength=self.num_known_classes)
        return counts[:self.num_known_classes]


# =============================================================================
# Open-Set Data Splitter
# =============================================================================

class OpenSetSplitter:
    """
    Split dataset into known/unknown classes for open-set recognition.

    Supports various splitting protocols:
    - random: Random split of classes
    - head_known: Head classes are known, tail classes are unknown
    - tail_known: Tail classes are known, head classes are unknown (harder)
    - stratified: Ensure both known/unknown have similar distributions
    """

    def __init__(
        self,
        dataset: Dataset,
        num_known_classes: int,
        split_protocol: str = "random",
        seed: int = 42,
    ):
        """
        Args:
            dataset: Full dataset
            num_known_classes: Number of classes to keep as known
            split_protocol: Splitting protocol ("random", "head_known", "tail_known")
            seed: Random seed
        """
        self.dataset = dataset
        self.num_known_classes = num_known_classes
        self.split_protocol = split_protocol
        self.seed = seed

        # Get all labels and classes
        if hasattr(dataset, 'labels'):
            self.all_labels = dataset.labels
        else:
            self.all_labels = np.array([dataset[i][1] for i in range(len(dataset))])

        self.all_classes = np.unique(self.all_labels)
        self.total_classes = len(self.all_classes)

        if num_known_classes >= self.total_classes:
            raise ValueError(f"num_known_classes ({num_known_classes}) must be < total_classes ({self.total_classes})")

        print(f"[OpenSetSplitter] Splitting dataset:")
        print(f"  - Total classes: {self.total_classes}")
        print(f"  - Known classes: {num_known_classes}")
        print(f"  - Unknown classes: {self.total_classes - num_known_classes}")
        print(f"  - Protocol: {split_protocol}")

    def split(self) -> Tuple[List[int], List[int], Dict[int, int]]:
        """
        Split classes into known/unknown.

        Returns:
            known_classes: List of known class IDs
            unknown_classes: List of unknown class IDs
            label_map: Mapping from original label to new label [0, num_known-1]
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Count samples per class
        class_counts = {c: (self.all_labels == c).sum() for c in self.all_classes}

        if self.split_protocol == "random":
            # Random split
            shuffled = list(self.all_classes)
            random.shuffle(shuffled)
            known_classes = shuffled[:self.num_known_classes]
            unknown_classes = shuffled[self.num_known_classes:]

        elif self.split_protocol == "head_known":
            # Head classes (more samples) are known
            sorted_classes = sorted(self.all_classes, key=lambda c: class_counts[c], reverse=True)
            known_classes = sorted_classes[:self.num_known_classes]
            unknown_classes = sorted_classes[self.num_known_classes:]

        elif self.split_protocol == "tail_known":
            # Tail classes (fewer samples) are known (harder scenario)
            sorted_classes = sorted(self.all_classes, key=lambda c: class_counts[c])
            known_classes = sorted_classes[:self.num_known_classes]
            unknown_classes = sorted_classes[self.num_known_classes:]

        elif self.split_protocol == "stratified":
            # Ensure known/unknown have similar sample distributions
            sorted_classes = sorted(self.all_classes, key=lambda c: class_counts[c], reverse=True)
            known_classes = []
            unknown_classes = []
            for i, c in enumerate(sorted_classes):
                if i % 2 == 0 and len(known_classes) < self.num_known_classes:
                    known_classes.append(c)
                elif i % 2 == 1 and len(known_classes) < self.num_known_classes:
                    known_classes.append(c)
                else:
                    unknown_classes.append(c)
            if len(known_classes) < self.num_known_classes:
                # Fill remaining
                remaining = self.num_known_classes - len(known_classes)
                known_classes.extend(unknown_classes[:remaining])
                unknown_classes = unknown_classes[remaining:]

        else:
            raise ValueError(f"Unknown split protocol: {self.split_protocol}")

        # Create label map: original -> new [0, num_known-1]
        known_classes_sorted = sorted(known_classes)
        label_map = {int(old): int(new) for new, old in enumerate(known_classes_sorted)}

        print(f"  - Known classes: {known_classes_sorted}")
        print(f"  - Unknown classes: {sorted(unknown_classes)}")

        return known_classes_sorted, sorted(unknown_classes), label_map

    def create_datasets(
        self,
        train_ratio: float = 0.8,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/val/test datasets for open-set recognition.

        Args:
            train_ratio: Ratio of known class samples for training

        Returns:
            train_dataset: Training set (known classes only)
            val_dataset: Validation set (known + unknown classes)
            test_dataset: Test set (known + unknown classes)
        """
        known_classes, unknown_classes, label_map = self.split()

        # Split indices
        known_indices = [i for i, label in enumerate(self.all_labels) if label in known_classes]
        unknown_indices = [i for i, label in enumerate(self.all_labels) if label in unknown_classes]

        # Split known into train/val
        random.shuffle(known_indices)
        split_idx = int(len(known_indices) * train_ratio)
        train_indices = known_indices[:split_idx]
        val_known_indices = known_indices[split_idx:]

        # Split unknown into val/test (50-50)
        random.shuffle(unknown_indices)
        split_idx = len(unknown_indices) // 2
        val_unknown_indices = unknown_indices[:split_idx]
        test_unknown_indices = unknown_indices[split_idx:]

        # Create datasets
        train_dataset = Subset(self.dataset, train_indices)
        train_dataset.labels = self.all_labels[train_indices]
        # Remap labels
        train_dataset.labels = np.array([label_map[int(l)] for l in train_dataset.labels])
        train_dataset.num_classes = self.num_known_classes
        train_dataset.class_counts = np.bincount(train_dataset.labels, minlength=self.num_known_classes)

        val_known_dataset = Subset(self.dataset, val_known_indices)
        val_unknown_dataset = Subset(self.dataset, val_unknown_indices)

        test_known_dataset = Subset(self.dataset, val_known_indices)  # Reuse val known
        test_unknown_dataset = Subset(self.dataset, test_unknown_indices)

        # Wrap in OpenSetDataset
        val_dataset = OpenSetDataset(val_known_dataset, val_unknown_dataset, label_map)
        test_dataset = OpenSetDataset(test_known_dataset, test_unknown_dataset, label_map)

        print(f"\n[Dataset Split]")
        print(f"  - Train (known only): {len(train_dataset)} samples")
        print(f"  - Val (known + unknown): {len(val_dataset)} samples")
        print(f"  - Test (known + unknown): {len(test_dataset)} samples")

        return train_dataset, val_dataset, test_dataset


# =============================================================================
# Long-Tail Open-Set Data Loader Factory
# =============================================================================

def create_longtail_openset_dataloaders(
    data_path: str,
    num_known_classes: int,
    split_protocol: str = "random",
    imbalance_ratio: float = 100.0,
    batch_size: int = 128,
    num_workers: int = 4,
    target_length: int = 4800,
    normalize: bool = True,
    augmentation: bool = True,
    sampling_strategy: str = "none",
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create dataloaders for long-tail open-set recognition.

    Args:
        data_path: Path to data file
        num_known_classes: Number of known classes
        split_protocol: Open-set splitting protocol
        imbalance_ratio: Long-tail imbalance ratio
        batch_size: Batch size
        num_workers: Number of workers
        target_length: Target sequence length
        normalize: Whether to normalize
        augmentation: Whether to use data augmentation
        sampling_strategy: Sampling strategy for long-tail
        seed: Random seed
        **kwargs: Additional arguments

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        info: Dictionary with dataset information
    """
    print("\n" + "=" * 80)
    print("Creating Long-Tail Open-Set DataLoaders")
    print("=" * 80)

    # Load full dataset
    transforms_train = []
    transforms_val = []

    if augmentation:
        transforms_train.extend([
            RandomTimeShift(max_shift=100, p=0.5),
            RandomAmplitude(p=0.5),
            RandomNoise(p=0.5),
        ])

    if normalize:
        transforms_train.append(PerSampleNormalize())
        transforms_val.append(PerSampleNormalize())

    transforms_train.insert(0, ToTensor())
    transforms_val.insert(0, ToTensor())

    transform_train = Compose(transforms_train)
    transform_val = Compose(transforms_val)

    # Load dataset
    full_dataset = SignalDataset(
        data_path=data_path,
        target_length=target_length,
        transform=None,  # Will apply later
        normalize=False,  # Will apply later
        **kwargs,
    )

    # Split into known/unknown
    splitter = OpenSetSplitter(
        dataset=full_dataset,
        num_known_classes=num_known_classes,
        split_protocol=split_protocol,
        seed=seed,
    )

    train_dataset, val_dataset, test_dataset = splitter.create_datasets(train_ratio=0.8)

    # Apply transforms
    train_dataset.dataset.transform = transform_train
    val_dataset.known_dataset.dataset.transform = transform_val
    if val_dataset.unknown_dataset is not None:
        val_dataset.unknown_dataset.dataset.transform = transform_val
    test_dataset.known_dataset.dataset.transform = transform_val
    if test_dataset.unknown_dataset is not None:
        test_dataset.unknown_dataset.dataset.transform = transform_val

    # Create long-tail distribution for training set
    if imbalance_ratio > 1.0:
        print(f"\n[Creating Long-Tail Distribution]")
        print(f"  - Imbalance ratio: {imbalance_ratio}")

        # Compute target counts (exponential decay)
        num_classes = num_known_classes
        max_count = len(train_dataset) // num_classes
        min_count = int(max_count / imbalance_ratio)

        # Exponential decay
        mu = np.log(imbalance_ratio) / (num_classes - 1)
        class_counts_target = max_count * np.exp(-mu * np.arange(num_classes))
        class_counts_target = np.maximum(class_counts_target, min_count).astype(int)

        # Subsample training set to match target distribution
        new_indices = []
        for c in range(num_classes):
            class_indices = np.where(train_dataset.labels == c)[0]
            if len(class_indices) > class_counts_target[c]:
                class_indices = np.random.choice(class_indices, class_counts_target[c], replace=False)
            new_indices.extend(class_indices)

        # Create new training dataset
        original_indices = train_dataset.indices
        new_absolute_indices = [original_indices[i] for i in new_indices]

        train_dataset = Subset(train_dataset.dataset, new_absolute_indices)
        train_dataset.labels = np.array([splitter.label_map[int(full_dataset.labels[i])] for i in new_absolute_indices])
        train_dataset.num_classes = num_known_classes
        train_dataset.class_counts = np.bincount(train_dataset.labels, minlength=num_known_classes)

        print(f"  - New train size: {len(train_dataset)}")
        print(f"  - Class counts: min={train_dataset.class_counts.min()}, max={train_dataset.class_counts.max()}")

    # Create samplers
    sampler_train = get_sampler(
        labels=train_dataset.labels,
        num_classes=train_dataset.num_classes,
        strategy=sampling_strategy,
        **kwargs,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Collect info
    info = {
        "num_known_classes": num_known_classes,
        "num_unknown_classes": splitter.total_classes - num_known_classes,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "class_counts": train_dataset.class_counts,
        "imbalance_ratio": train_dataset.class_counts.max() / max(train_dataset.class_counts.min(), 1),
        "label_map": splitter.split()[2],
    }

    print("\n" + "=" * 80)
    print(f"DataLoaders created successfully!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader, info
