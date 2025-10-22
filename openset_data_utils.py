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
    ADSBSignalDataset,
    make_sampler,
    Compose,
    ToTensor,
    RandomTimeShift,
    RandomAmplitude,
    RandomGaussianNoise,
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
# Known-Class Training Dataset
# =============================================================================

class KnownClassSubset(Dataset):
    """Subset wrapper that exposes remapped labels for known-class training."""

    def __init__(
        self,
        base_subset: Subset,
        remapped_labels: np.ndarray,
        num_classes: int,
    ):
        if len(base_subset) != len(remapped_labels):
            raise ValueError(
                "base_subset and remapped_labels must have the same length"
            )

        self._subset = base_subset
        self.labels = np.asarray(remapped_labels, dtype=np.int64)
        self.num_classes = int(num_classes)
        if self.num_classes < 0:
            raise ValueError("num_classes must be non-negative")

        if len(self.labels) == 0:
            self.class_counts = np.zeros(self.num_classes, dtype=np.int64)
        else:
            self.class_counts = np.bincount(
                self.labels, minlength=self.num_classes
            )

        # Expose underlying indices for reproducibility/debugging if available
        self.indices = getattr(base_subset, "indices", None)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._subset)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._subset[idx]
        return x, int(self.labels[idx])

    @property
    def dataset(self):
        return self._subset.dataset


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

        # Store as instance attributes for later use
        self.known_classes = known_classes_sorted
        self.unknown_classes = sorted(unknown_classes)
        self.label_map = label_map

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
    alpha: float = 0.5,
    alpha_start: float = 0.5,
    alpha_end: float = 0.0,
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
        alpha: Alpha for power sampling
        alpha_start: Starting alpha for progressive power sampling
        alpha_end: Ending alpha for progressive power sampling
        **kwargs: Additional arguments for dataset

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        info: Dictionary with dataset information
    """
    print("\n" + "=" * 80)
    print("Creating Long-Tail Open-Set DataLoaders")
    print("=" * 80)

    # Separate sampler parameters from dataset parameters
    sampler_params = {
        'seed': seed,
        'alpha': alpha,
        'alpha_start': alpha_start,
        'alpha_end': alpha_end,
    }

    # Load full dataset
    transforms_train = []
    transforms_val = []

    if augmentation:
        transforms_train.extend([
            RandomTimeShift(max_shift=100, p=0.5),
            RandomAmplitude(p=0.5),
            RandomGaussianNoise(p=0.5),
        ])

    if normalize:
        transforms_train.append(PerSampleNormalize())
        transforms_val.append(PerSampleNormalize())

    transforms_train.insert(0, ToTensor())
    transforms_val.insert(0, ToTensor())

    transform_train = Compose(transforms_train)
    transform_val = Compose(transforms_val)

    # Load dataset (only pass supported parameters)
    # Extract dataset-specific parameters from kwargs
    dataset_params = {
        'split': kwargs.get('split', None),
        'data_key': kwargs.get('data_key', None),
        'label_key': kwargs.get('label_key', None),
        'in_memory': kwargs.get('in_memory', False),
    }
    # Remove None values
    dataset_params = {k: v for k, v in dataset_params.items() if v is not None}

    full_dataset = ADSBSignalDataset(
        path=data_path,
        target_length=target_length,
        transforms=None,  # Will apply later
        normalize=False,  # Will apply later
        seed=seed,
        **dataset_params,  # Only dataset-specific parameters
    )

    # Split into known/unknown
    splitter = OpenSetSplitter(
        dataset=full_dataset,
        num_known_classes=num_known_classes,
        split_protocol=split_protocol,
        seed=seed,
    )

    train_subset, val_dataset, test_dataset = splitter.create_datasets(train_ratio=0.8)

    train_indices = np.asarray(
        getattr(train_subset, "indices", np.arange(len(train_subset))),
        dtype=np.int64,
    )
    train_labels_original = splitter.all_labels[train_indices]
    remapped_train_labels = np.array(
        [splitter.label_map[int(l)] for l in train_labels_original],
        dtype=np.int64,
    )

    apply_long_tail = imbalance_ratio > 1.0 and len(remapped_train_labels) > 0
    if apply_long_tail:
        print(f"\n[Creating Long-Tail Distribution]")
        print(f"  - Imbalance ratio: {imbalance_ratio}")

        rng = np.random.default_rng(seed)
        class_counts_actual = np.bincount(
            remapped_train_labels, minlength=num_known_classes
        )
        max_count = int(class_counts_actual.max())
        min_count = max(int(max_count / imbalance_ratio), 1)

        if num_known_classes > 1:
            mu = np.log(imbalance_ratio) / (num_known_classes - 1)
            class_counts_target = np.round(
                max_count * np.exp(-mu * np.arange(num_known_classes))
            ).astype(int)
        else:
            class_counts_target = np.array([max_count], dtype=int)

        class_counts_target = np.maximum(class_counts_target, min_count)
        class_counts_target = np.minimum(class_counts_target, class_counts_actual)
        class_counts_target = np.where(
            class_counts_actual == 0, 0, class_counts_target
        )

        selected_positions: List[np.ndarray] = []
        for c in range(num_known_classes):
            class_positions = np.where(remapped_train_labels == c)[0]
            target = int(class_counts_target[c])
            if target <= 0 or len(class_positions) == 0:
                continue
            if len(class_positions) > target:
                chosen = rng.choice(class_positions, target, replace=False)
            else:
                chosen = class_positions
            selected_positions.append(np.asarray(chosen, dtype=np.int64))

        if selected_positions:
            selected_positions = np.concatenate(selected_positions)
            rng.shuffle(selected_positions)
            train_indices = train_indices[selected_positions]
            remapped_train_labels = remapped_train_labels[selected_positions]
        else:
            print(
                "  - Warning: unable to construct long-tail schedule; retaining original training set."
            )

    train_subset = Subset(splitter.dataset, train_indices.tolist())

    # Apply transforms
    train_subset.dataset.transforms = transform_train
    val_dataset.known_dataset.dataset.transforms = transform_val
    if val_dataset.unknown_dataset is not None:
        val_dataset.unknown_dataset.dataset.transforms = transform_val
    test_dataset.known_dataset.dataset.transforms = transform_val
    if test_dataset.unknown_dataset is not None:
        test_dataset.unknown_dataset.dataset.transforms = transform_val

    train_dataset = KnownClassSubset(
        train_subset,
        remapped_train_labels,
        num_known_classes,
    )

    if apply_long_tail and train_dataset.class_counts.size > 0:
        print(f"  - New train size: {len(train_dataset)}")
        print(
            "  - Class counts: min={}, max={}".format(
                int(train_dataset.class_counts.min()),
                int(train_dataset.class_counts.max()),
            )
        )

    # Create samplers
    sampler_train = make_sampler(
        labels=train_dataset.labels,
        method=sampling_strategy,
        **sampler_params,  # Pass sampler-specific parameters
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
        "label_map": splitter.label_map,  # Use stored attribute instead of calling split() again
    }

    print("\n" + "=" * 80)
    print(f"DataLoaders created successfully!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader, info
