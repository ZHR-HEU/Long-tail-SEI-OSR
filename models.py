# -*- coding: utf-8 -*-
"""
Complete collection of neural network models for imbalanced learning (Enhanced).

This module contains various neural network architectures and classification heads
specifically designed for handling imbalanced datasets, with a focus on signal
processing tasks (e.g., I/Q baseband).

What’s new vs. baseline:
- Bias-free Conv1d when followed by normalization (waste-free with BN/GN/LN)
- Normalization factory (auto BN→GN/LN fallback)
- Cosine-margin head (LDAM-ready) and Logit-Adjusted linear head
- Robust frequency-domain feature builder (RI / Mag-Phase / Log-Power)
- Mixture-of-Experts load-balance regularizer
- Cleaner initialization utilities with log-prior bias option
- Minor API helpers: swap_classifier(), moe_load_balance_loss(), etc.

Dependencies: torch, numpy
Author: Enhanced Implementation
Date: 2025
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Normalization & Small Utilities
# =============================================================================

def Norm1d(num_channels: int, kind: str = "auto", groups: int = 8) -> nn.Module:
    """
    Normalization factory for 1D features.
    kind: "auto" | "bn" | "gn" | "ln"
    - "auto": default to BatchNorm1d (change to "gn" if your batch is tiny)
    - "gn": GroupNorm with up to `groups` groups
    - "ln": LayerNorm-equivalent for 1D (GN with groups=1)
    """
    kind = (kind or "auto").lower()
    if kind == "auto":
        kind = "bn"
    if kind == "bn":
        return nn.BatchNorm1d(num_channels)
    if kind == "gn":
        g = min(groups, num_channels)
        return nn.GroupNorm(g, num_channels)
    if kind == "ln":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unknown norm kind: {kind}")


# =============================================================================
# Enhanced Model Initialization Utilities
# =============================================================================

class EnhancedClassifierInitializer:
    """
    Enhanced classifier initializer supporting multiple strategies.

    Args:
        num_classes (int): Number of classes in the dataset.
        class_counts (np.ndarray): Number of samples per class.
    """

    def __init__(self, num_classes: int, class_counts: np.ndarray):
        self.num_classes = int(num_classes)
        self.class_counts = np.asarray(class_counts, dtype=np.float64).reshape(-1)
        total = np.clip(self.class_counts.sum(), 1.0, None)
        self.class_frequencies = self.class_counts / total
        print(f"[INFO] ClassifierInitializer for {self.num_classes} classes")
        print(f"  - Class freq range: {self.class_frequencies.min():.6f} ~ {self.class_frequencies.max():.6f}")

    def frequency_aware_bias_init(self, classifier: nn.Linear, alpha: float = 0.1):
        """
        Legacy frequency-aware bias: -log(freq/max)*alpha
        """
        max_freq = float(np.max(self.class_frequencies))
        bias_init = -np.log(self.class_frequencies / max_freq + 1e-12) * float(alpha)
        with torch.no_grad():
            if classifier.bias is not None:
                classifier.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
        print(f"  - Applied frequency-aware bias init (alpha={alpha})")

    def set_bias_to_log_prior(self, classifier: nn.Linear, temperature: float = 1.0):
        """
        Set bias to log-prior / T (Menon et al., Logit Adjustment). Useful as a calibrated prior.
        """
        prior = self.class_frequencies
        with torch.no_grad():
            if classifier.bias is not None:
                classifier.bias.copy_(torch.tensor(np.log(prior + 1e-12) / max(1e-6, temperature),
                                                   dtype=torch.float32))
        print(f"  - Set classifier bias to log-prior / T (T={temperature})")

    def balanced_xavier_init(self, classifier: nn.Linear, use_log_prior_bias: bool = True, temperature: float = 1.0):
        nn.init.xavier_uniform_(classifier.weight)
        if use_log_prior_bias:
            self.set_bias_to_log_prior(classifier, temperature=temperature)
        else:
            self.frequency_aware_bias_init(classifier, alpha=0.1)

    def he_kaiming_init(self, classifier: nn.Linear, use_log_prior_bias: bool = True, temperature: float = 1.0):
        nn.init.kaiming_uniform_(classifier.weight, mode='fan_in', nonlinearity='relu')
        if use_log_prior_bias:
            self.set_bias_to_log_prior(classifier, temperature=temperature)
        else:
            self.frequency_aware_bias_init(classifier, alpha=0.15)


class TemperatureScaledClassifier(nn.Module):
    """
    Classifier with learnable temperature scaling.
    Args:
        in_features (int)
        num_classes (int)
        initial_temperature (float)
    """

    def __init__(self, in_features: int, num_classes: int, initial_temperature: float = 1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.temperature = nn.Parameter(torch.tensor(float(initial_temperature), dtype=torch.float32))
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits / (self.temperature + 1e-8)


# =============================================================================
# Classification Heads for Imbalance
# =============================================================================

class CosineMarginClassifier(nn.Module):
    """
    Cosine classifier with optional per-class margin (LDAM style).
    Forward:
        logits = s * cos(theta); if y provided and margins is not None -> subtract margin on targets
    Args:
        in_features, num_classes, scale s, margins (Optional[np.ndarray])
    """
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0,
                 margins: Optional[np.ndarray] = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.scale = float(scale)
        if margins is None:
            self.register_buffer("margins", None, persistent=False)
        else:
            m = torch.as_tensor(margins, dtype=torch.float32)
            self.register_buffer("margins", m, persistent=False)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        logits = torch.matmul(x_norm, w_norm.t())
        if y is not None and self.margins is not None:
            margins = self.margins.to(logits.device)
            logits = logits.clone()
            logits.scatter_add_(1, y.view(-1, 1), (-margins[y]).view(-1, 1))
        return self.scale * logits


def ldam_margins_from_counts(class_counts: np.ndarray, power: float = 0.25, max_m: float = 0.5) -> np.ndarray:
    """
    LDAM margin: m_c ∝ 1 / n_c^{power} (default power=1/4), then rescaled to max_m.
    """
    cc = np.clip(class_counts.astype(np.float32), 1.0, None)
    raw = 1.0 / (cc ** float(power))
    raw = raw / raw.max() * float(max_m)
    return raw


class LogitAdjustedLinear(nn.Module):
    """
    Linear head with logit adjustment at run time:
        logits = xW^T + b - tau * log(prior)
    Ref: Menon et al., "Long-Tail Learning via Logit Adjustment"
    """
    def __init__(self, in_features: int, num_classes: int,
                 class_counts: np.ndarray, tau: float = 1.0, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=bias)
        prior = class_counts.astype(np.float64)
        prior = prior / np.clip(prior.sum(), 1.0, None)
        self.register_buffer("log_prior", torch.tensor(np.log(prior + 1e-12), dtype=torch.float32),
                             persistent=False)
        self.tau = float(tau)
        if bias:
            with torch.no_grad():
                self.linear.bias.copy_(self.log_prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits - self.tau * self.log_prior


def swap_classifier(model: nn.Module, head: str,
                    class_counts: Optional[np.ndarray] = None,
                    *, scale: float = 30.0, tau: float = 1.0,
                    ldam_power: float = 0.25, ldam_max_m: float = 0.5) -> nn.Module:
    """
    Replace `model.classifier` with an advanced head.
    head in {"linear", "cosine", "cosine_ldam", "logit_adjust"}
    """
    if not hasattr(model, "classifier"):
        raise AttributeError("Model has no attribute 'classifier'")
    old: nn.Module = model.classifier  # type: ignore
    if isinstance(old, nn.Linear):
        in_features = old.in_features
        num_classes = old.out_features
    else:
        # Try to deduce from a dummy input if possible
        raise TypeError("swap_classifier expects a linear classifier attribute on the model")

    if head == "linear":
        new_head = nn.Linear(in_features, num_classes, bias=True)
    elif head == "cosine":
        new_head = CosineMarginClassifier(in_features, num_classes, scale=scale, margins=None)
    elif head == "cosine_ldam":
        if class_counts is None:
            raise ValueError("cosine_ldam requires class_counts")
        margins = ldam_margins_from_counts(class_counts, power=ldam_power, max_m=ldam_max_m)
        new_head = CosineMarginClassifier(in_features, num_classes, scale=scale, margins=margins)
    elif head == "logit_adjust":
        if class_counts is None:
            raise ValueError("logit_adjust requires class_counts")
        new_head = LogitAdjustedLinear(in_features, num_classes, class_counts, tau=tau, bias=True)
    else:
        raise ValueError(f"Unknown head: {head}")

    model.classifier = new_head  # type: ignore
    return model


# =============================================================================
# Frequency-Domain Feature Builder
# =============================================================================

class SpectralFeatures(nn.Module):
    """
    Compute frequency-domain features from I/Q.
    mode: "ri" | "magphase" | "powerlog"
    - "ri": real & imag as 2 channels
    - "magphase": magnitude and phase
    - "powerlog": log(1 + power) replicated to 2 channels (for 2-ch backbones)
    """
    def __init__(self, mode: str = "powerlog", window: Optional[str] = "hann"):
        super().__init__()
        self.mode = (mode or "powerlog").lower()
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, T] (I/Q)
        xc = torch.complex(x[:, 0, :], x[:, 1, :])  # [B, T]
        if self.window is not None:
            if self.window.lower() == "hann":
                w = torch.hann_window(xc.shape[-1], device=xc.device, dtype=xc.real.dtype)
            else:
                raise ValueError(f"Unknown window: {self.window}")
            xc = xc * w
        X = torch.fft.fft(xc, dim=-1, norm="ortho")
        if self.mode == "ri":
            feat = torch.stack([X.real, X.imag], dim=1)
        elif self.mode == "magphase":
            mag = X.abs()
            phs = torch.atan2(X.imag, X.real)
            feat = torch.stack([mag, phs], dim=1)
        elif self.mode == "powerlog":
            p = (X.real ** 2 + X.imag ** 2)
            feat = torch.log1p(p).unsqueeze(1).repeat(1, 2, 1)
        else:
            raise ValueError(f"Unknown spectral mode: {self.mode}")
        return feat  # [B, 2, F]


# =============================================================================
# Basic Building Blocks
# =============================================================================

class BasicBlock1D(nn.Module):
    """
    Basic residual block for 1D convolutions.
    Args:
        in_ch, out_ch, stride, dilation, norm_kind
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1, norm_kind: str = "auto"):
        super().__init__()
        pad = ((3 - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1 = Norm1d(out_ch, kind=norm_kind)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2 = Norm1d(out_ch, kind=norm_kind)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                Norm1d(out_ch, kind=norm_kind)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.act(out + identity)
        return out


class DilatedTCNBlock(nn.Module):
    """
    Dilated Temporal Convolutional Network block.
    """
    def __init__(self, ch: int, dilation: int, norm_kind: str = "auto"):
        super().__init__()
        pad = dilation
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            Norm1d(ch, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            Norm1d(ch, kind=norm_kind)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return self.relu(out + x)


class ChannelAttention(nn.Module):
    """
    Channel attention module for feature refinement.
    """
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        mid = max(1, in_ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# =============================================================================
# Base Model Architectures
# =============================================================================

class ConvNetADSB(nn.Module):
    """
    Deep Convolutional Network for ADS-B signal classification.
    Args:
        num_classes, dropout_rate, use_attention, norm_kind
    """
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.1,
                 use_attention: bool = False, norm_kind: str = "auto"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_attention = bool(use_attention)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 50, kernel_size=15, stride=1, padding=7, bias=False),
            Norm1d(50, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            # Block 2
            nn.Conv1d(50, 100, kernel_size=11, stride=1, padding=5, bias=False),
            Norm1d(100, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),

            # Block 3
            nn.Conv1d(100, 150, kernel_size=7, stride=1, padding=3, bias=False),
            Norm1d(150, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            # Block 4
            nn.Conv1d(150, 200, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(200, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(200, 200, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(200, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            # Block 5
            nn.Conv1d(200, 300, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(300, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(300, 300, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(300, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            # Block 6
            nn.Conv1d(300, 350, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(350, 350, kernel_size=5, stride=1, padding=2, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            # Block 7
            nn.Conv1d(350, 350, kernel_size=7, stride=1, padding=3, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(350, 350, kernel_size=7, stride=1, padding=3, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            # Block 8
            nn.Conv1d(350, 350, kernel_size=7, stride=1, padding=3, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv1d(350, 350, kernel_size=7, stride=1, padding=3, bias=False),
            Norm1d(350, kind=norm_kind),
            nn.ReLU(inplace=True),
        )

        self.attention = ChannelAttention(350) if self.use_attention else None
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(350, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 2, T]
        features = self.features(x)
        if self.attention is not None:
            features = self.attention(features)
        pooled = self.global_pool(features).view(features.size(0), -1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled


class ResNet1D(nn.Module):
    """
    1D ResNet for signal processing tasks.
    Args:
        in_ch, base, num_blocks, num_classes, dropout_rate, use_attention, norm_kind
    """
    def __init__(self, in_ch: int = 2, base: int = 32, num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
                 num_classes: int = 10, dropout_rate: float = 0.1, use_attention: bool = False,
                 norm_kind: str = "auto"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_attention = bool(use_attention)
        self.norm_kind = norm_kind

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False),
            Norm1d(base, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Layers
        c = base
        self.layer1 = self._make_layer(c, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(c, c * 2, num_blocks[1], stride=2)
        c *= 2
        self.layer3 = self._make_layer(c, c * 2, num_blocks[2], stride=2)
        c *= 2
        self.layer4 = self._make_layer(c, c * 2, num_blocks[3], stride=2)
        c *= 2

        self.attention = ChannelAttention(c) if self.use_attention else None
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(c, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock1D(in_ch, out_ch, stride=stride, norm_kind=self.norm_kind)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, norm_kind=self.norm_kind))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        if self.attention is not None:
            h = self.attention(h)
        pooled = self.global_pool(h).squeeze(-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled


class DilatedTCN(nn.Module):
    """
    Dilated Temporal Convolutional Network.
    Args:
        in_ch, ch, num_layers, num_classes, dropout_rate, use_attention, norm_kind
    """
    def __init__(self, in_ch: int = 2, ch: int = 64, num_layers: int = 4, num_classes: int = 10,
                 dropout_rate: float = 0.1, use_attention: bool = False, norm_kind: str = "auto"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_attention = bool(use_attention)
        self.norm_kind = norm_kind

        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=1, bias=False),
            Norm1d(ch, kind=norm_kind),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(*[
            DilatedTCNBlock(ch, dilation=2 ** i, norm_kind=norm_kind) for i in range(num_layers)
        ])

        self.attention = ChannelAttention(ch) if self.use_attention else None
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(ch, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        h = self.blocks(h)
        if self.attention is not None:
            h = self.attention(h)
        pooled = self.global_pool(h).squeeze(-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, pooled


# =============================================================================
# Frequency Domain Models (using SpectralFeatures)
# =============================================================================

class FrequencyDomainExpert(nn.Module):
    """
    Frequency domain expert using ConvNetADSB backbone with robust spectral features.
    Args:
        num_classes, backbone_dropout, use_attention, spectral_mode, spectral_window, norm_kind
    """
    def __init__(self, num_classes: int = 10, backbone_dropout: float = 0.1,
                 use_attention: bool = False, spectral_mode: str = "powerlog",
                 spectral_window: Optional[str] = "hann", norm_kind: str = "auto"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.spec = SpectralFeatures(mode=spectral_mode, window=spectral_window)
        self.backbone = ConvNetADSB(num_classes=num_classes,
                                    dropout_rate=backbone_dropout,
                                    use_attention=use_attention,
                                    norm_kind=norm_kind)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 2, T]
        feat = self.spec(x)  # [B, 2, F]
        logits, pooled = self.backbone(feat)
        return logits, pooled


class ResNetFrequencyExpert(nn.Module):
    """
    Frequency domain expert using ResNet1D backbone with robust spectral features.
    """
    def __init__(self, base: int = 16, num_classes: int = 10, use_attention: bool = False,
                 spectral_mode: str = "powerlog", spectral_window: Optional[str] = "hann",
                 norm_kind: str = "auto"):
        super().__init__()
        self.spec = SpectralFeatures(mode=spectral_mode, window=spectral_window)
        self.backbone = ResNet1D(in_ch=2, base=base, num_blocks=(2, 2, 2, 2),
                                 num_classes=num_classes, use_attention=use_attention,
                                 norm_kind=norm_kind)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.spec(x)
        logits, pooled = self.backbone(feat)
        return logits, pooled


# =============================================================================
# Mixture of Experts Models + Load-Balance Regularizer
# =============================================================================

def moe_load_balance_loss(gate_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Encourage using all experts: maximize entropy of average gate probs.
    LB = - H(mean_b(gate)) = sum_i p_i log p_i
    """
    p = gate_weights.mean(dim=0)  # [E]
    return (p * (p + eps).log()).sum()


class MixtureOfExpertsConvNet(nn.Module):
    """
    Mixture of Experts model based on ConvNetADSB + DilatedTCN + Frequency expert.
    Gating: light conv trunk + MLP -> softmax probs over experts.
    """
    def __init__(self, num_classes: int = 10, gate_hidden: int = 64, expert_dropout: float = 0.1,
                 dropout_rate: Optional[float] = None, use_attention: bool = False, norm_kind: str = "auto"):
        super().__init__()
        if dropout_rate is not None:
            expert_dropout = dropout_rate
        self.num_classes = int(num_classes)
        self.num_experts = 3

        # Experts
        self.expert_conv_time = ConvNetADSB(num_classes=num_classes,
                                            dropout_rate=expert_dropout,
                                            use_attention=use_attention,
                                            norm_kind=norm_kind)
        self.expert_tcn = DilatedTCN(in_ch=2, ch=64, num_layers=4, num_classes=num_classes,
                                     dropout_rate=expert_dropout, use_attention=use_attention,
                                     norm_kind=norm_kind)
        self.expert_conv_freq = FrequencyDomainExpert(num_classes=num_classes,
                                                      backbone_dropout=expert_dropout,
                                                      use_attention=use_attention,
                                                      spectral_mode="powerlog",
                                                      spectral_window="hann",
                                                      norm_kind=norm_kind)

        # Gating network
        self.gate_stem = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            Norm1d(16, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            Norm1d(32, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(32, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, self.num_experts)
        )

        self._initialize_gate_weights()

    def _initialize_gate_weights(self):
        for m in self.gate_stem.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_features = self.gate_stem(x).squeeze(-1)  # [B, 32]
        gate_logits = self.gate_mlp(gate_features)  # [B, E]
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, E]

        logit1, _ = self.expert_conv_time(x)
        logit2, _ = self.expert_tcn(x)
        logit3, _ = self.expert_conv_freq(x)

        logits = (gate_weights[:, 0:1] * logit1 +
                  gate_weights[:, 1:2] * logit2 +
                  gate_weights[:, 2:3] * logit3)
        return logits, gate_weights


class MixtureOfExpertsResNet(nn.Module):
    """
    Mixture of Experts model based on ResNet1D + DilatedTCN + ResNet frequency expert.
    """
    def __init__(self, num_classes: int = 10, gate_hidden: int = 64, expert_dropout: float = 0.1,
                 dropout_rate: Optional[float] = None, use_attention: bool = False, norm_kind: str = "auto"):
        super().__init__()
        if dropout_rate is not None:
            expert_dropout = dropout_rate
        self.num_classes = int(num_classes)
        self.num_experts = 3

        self.expert_time_resnet = ResNet1D(in_ch=2, base=32, num_blocks=(2, 2, 2, 2),
                                           num_classes=num_classes, dropout_rate=expert_dropout,
                                           use_attention=use_attention, norm_kind=norm_kind)
        self.expert_tcn = DilatedTCN(in_ch=2, ch=64, num_layers=4, num_classes=num_classes,
                                     dropout_rate=expert_dropout, use_attention=use_attention,
                                     norm_kind=norm_kind)
        self.expert_freq = ResNetFrequencyExpert(base=16, num_classes=num_classes,
                                                 use_attention=use_attention, norm_kind=norm_kind)

        self.gate_stem = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3, bias=False),
            Norm1d(16, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            Norm1d(32, kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(32, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, self.num_experts)
        )

        self._initialize_gate_weights()

    def _initialize_gate_weights(self):
        for m in self.gate_stem.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_features = self.gate_stem(x).squeeze(-1)
        gate_logits = self.gate_mlp(gate_features)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        logit1, _ = self.expert_time_resnet(x)
        logit2, _ = self.expert_tcn(x)
        logit3, _ = self.expert_freq(x)

        logits = (gate_weights[:, 0:1] * logit1 +
                  gate_weights[:, 1:2] * logit2 +
                  gate_weights[:, 2:3] * logit3)
        return logits, gate_weights


# =============================================================================
# Model Factory and Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, Any] = {
    # Base models
    'ConvNetADSB': ConvNetADSB,
    'ResNet1D': ResNet1D,
    'DilatedTCN': DilatedTCN,

    # Frequency domain experts
    'FrequencyDomainExpert': FrequencyDomainExpert,
    'ResNetFrequencyExpert': ResNetFrequencyExpert,

    # Mixture of experts models
    'MixtureOfExpertsConvNet': MixtureOfExpertsConvNet,
    'MixtureOfExpertsResNet': MixtureOfExpertsResNet,

    # Enhanced classifiers (heads)
    'TemperatureScaledClassifier': TemperatureScaledClassifier,
}


def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    For TemperatureScaledClassifier, pass in_features in kwargs.
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model: {model_name}. Available models: {available_models}")

    model_class = MODEL_REGISTRY[model_name]
    if model_name == 'TemperatureScaledClassifier':
        if 'in_features' not in kwargs:
            raise ValueError("TemperatureScaledClassifier requires 'in_features' parameter")
        return model_class(num_classes=num_classes, **kwargs)
    else:
        return model_class(num_classes=num_classes, **kwargs)


def get_model_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available models.
    """
    return {
        'Base Models': {
            'ConvNetADSB': 'Deep CNN for ADS-B signal classification (bias-free conv + flexible norm)',
            'ResNet1D': '1D ResNet for signal processing (bias-free conv + flexible norm)',
            'DilatedTCN': 'Dilated TCN (bias-free conv + flexible norm)'
        },
        'Frequency Domain': {
            'FrequencyDomainExpert': 'Frequency expert using ConvNetADSB + robust spectral features',
            'ResNetFrequencyExpert': 'Frequency expert using ResNet1D + robust spectral features'
        },
        'Mixture of Experts': {
            'MixtureOfExpertsConvNet': 'MoE with Conv/TCN/Freq experts + gating',
            'MixtureOfExpertsResNet': 'MoE with ResNet/TCN/Freq experts + gating'
        },
        'Enhanced Classifiers': {
            'TemperatureScaledClassifier': 'Classifier with learnable temperature scaling'
        }
    }


def list_available_models() -> List[str]:
    """
    List all available models.
    """
    return list(MODEL_REGISTRY.keys())


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """
    Count the number of trainable parameters in a model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        'trainable_parameters': float(trainable_params),
        'total_parameters': float(total_params),
        'trainable_percentage': (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    }


# =============================================================================
# Model Analysis Utilities
# =============================================================================

def analyze_model_complexity(model: nn.Module, input_shape: Tuple[int, int] = (2, 1024)) -> Dict[str, Any]:
    """
    Analyze model computational complexity (rough).
    Args:
        model, input_shape=(C, T)
    """
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    param_info = count_parameters(model)
    model_size_mb = param_info['total_parameters'] * 4.0 / (1024 ** 2)  # float32

    try:
        with torch.no_grad():
            if isinstance(model, TemperatureScaledClassifier):
                dummy_input = torch.randn(1, model.in_features)
            output = model(dummy_input)
            output_shape = output[0].shape if isinstance(output, tuple) else output.shape
        forward_pass_success = True
    except Exception as e:
        print(f"[analyze] Forward pass failed: {e}")
        output_shape = None
        forward_pass_success = False

    return {
        'model_name': model.__class__.__name__,
        'parameters': param_info,
        'model_size_mb': model_size_mb,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'forward_pass_success': forward_pass_success
    }


def init_model_for_imbalanced(model: nn.Module, class_counts: np.ndarray,
                              init_type: str = 'balanced_xavier',
                              use_log_prior_bias: bool = True,
                              temperature: float = 1.0) -> nn.Module:
    """
    Initialize model for imbalanced learning.
    init_type: 'balanced_xavier' | 'he_kaiming'
    use_log_prior_bias: if True, set bias to log-prior/T; else use frequency-aware bias
    """
    num_classes = int(len(class_counts))
    initializer = EnhancedClassifierInitializer(num_classes, class_counts)

    # Find the classifier layer
    classifier: Optional[nn.Linear] = None
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):  # type: ignore
        classifier = model.classifier  # type: ignore
    elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):  # type: ignore
        classifier = model.linear  # type: ignore
    else:
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                classifier = module
        # May find the last linear, not always the classifier, but good fallback.

    if classifier is not None:
        if init_type == 'balanced_xavier':
            initializer.balanced_xavier_init(classifier, use_log_prior_bias=use_log_prior_bias, temperature=temperature)
        elif init_type == 'he_kaiming':
            initializer.he_kaiming_init(classifier, use_log_prior_bias=use_log_prior_bias, temperature=temperature)
        else:
            print(f"[init] Unknown init_type '{init_type}', using balanced_xavier")
            initializer.balanced_xavier_init(classifier, use_log_prior_bias=use_log_prior_bias, temperature=temperature)
    else:
        print("[init] Warning: Could not find a Linear classifier layer for initialization")

    return model


# =============================================================================
# Usage (docstring)
# =============================================================================
"""
Examples:

1) Create a basic ConvNet model:
   model = create_model('ConvNetADSB', num_classes=8)

2) ResNet1D with custom params & attention:
   model = create_model('ResNet1D', num_classes=10, base=64, dropout_rate=0.2, use_attention=True)

3) Mixture of Experts:
   model = create_model('MixtureOfExpertsConvNet', num_classes=8, gate_hidden=128, use_attention=True)

4) Temperature-scaled classifier:
   model = create_model('TemperatureScaledClassifier', num_classes=10, in_features=512, initial_temperature=2.0)

5) Enhanced initialization (log-prior bias):
   class_counts = np.array([1000, 100, 50, 25, 10])
   model = create_model('ConvNetADSB', num_classes=5)
   model = init_model_for_imbalanced(model, class_counts, init_type='balanced_xavier',
                                     use_log_prior_bias=True, temperature=1.0)

6) Swap classifier head (cosine+LDAM, or logit adjustment):
   class_counts = np.array([1000, 100, 50, 25, 10])
   model = create_model('ResNet1D', num_classes=len(class_counts))
   swap_classifier(model, head="cosine_ldam", class_counts=class_counts, scale=32.0, ldam_power=0.25, ldam_max_m=0.5)
   # or:
   # swap_classifier(model, head="logit_adjust", class_counts=class_counts, tau=1.2)

7) MoE load-balance regularizer (training step):
   logits, gate = moe_model(x)
   ce = F.cross_entropy(logits, y)
   lb = 5e-3 * moe_load_balance_loss(gate)
   loss = ce + lb

8) Complexity analysis:
   analysis = analyze_model_complexity(model, input_shape=(2, 1024))
   print(analysis)

9) List available models:
   print(list_available_models())
   print(get_model_info())
"""
