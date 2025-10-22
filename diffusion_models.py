# -*- coding: utf-8 -*-
"""
Diffusion Models for Feature Space - Used for Open-Set Recognition (NOT Data Augmentation)

This module implements diffusion models in the feature space for:
1. Feature reconstruction and anomaly detection
2. Open-set sample detection via reconstruction error
3. Likelihood-based outlier detection
4. Feature space denoising for improved representations

Key Innovation: Diffusion models operate on learned features, not raw data,
enabling efficient open-set detection by measuring how well features fit
the known class distribution.

References:
- Denoising Diffusion Probabilistic Models (DDPM)
- Score-based Generative Models
- Diffusion Models for Anomaly Detection

Author: Enhanced Implementation for Long-Tail Open-Set Recognition
Date: 2025
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Beta Schedule for Diffusion Process
# =============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    More stable training than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_beta_schedule(schedule: str, timesteps: int, **kwargs) -> torch.Tensor:
    """Factory function for beta schedules."""
    if schedule == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    elif schedule == "cosine":
        return cosine_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


# =============================================================================
# Feature Space Diffusion Model
# =============================================================================

class FeatureDiffusion(nn.Module):
    """
    Diffusion model operating in feature space for open-set recognition.

    Unlike traditional diffusion models for generation, this model:
    1. Takes high-level features as input (not raw data)
    2. Learns to denoise features corrupted by Gaussian noise
    3. Uses reconstruction error as an anomaly score for open-set detection
    4. Can be conditioned on class labels for class-specific feature modeling

    Args:
        feature_dim: Dimension of input features
        hidden_dims: List of hidden dimensions for the denoising network
        timesteps: Number of diffusion timesteps
        beta_schedule: Beta schedule type ("linear" or "cosine")
        conditional: Whether to use class conditioning
        num_classes: Number of classes (required if conditional=True)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list = [512, 256, 512],
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        conditional: bool = True,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.timesteps = timesteps
        self.conditional = conditional
        self.num_classes = num_classes

        if conditional and num_classes is None:
            raise ValueError("num_classes must be specified when conditional=True")

        # Diffusion parameters
        betas = get_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance for denoising
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))

        # Time embedding
        # NOTE: the sinusoidal embedding implementation below assumes an even
        # dimensionality because it concatenates sine and cosine pairs of size
        # ``dim // 2``.  When ``feature_dim`` is not divisible by eight (e.g.
        # 348 -> 348 // 4 = 87), using ``feature_dim // 4`` would yield an odd
        # ``time_embed_dim`` which subsequently makes the embedding one element
        # smaller than expected and breaks the following linear layer.  We
        # therefore round up to the nearest even value and clamp to at least
        # two dimensions to keep the downstream layers well-defined.
        time_embed_dim = max(2, feature_dim // 4)
        if time_embed_dim % 2 == 1:
            time_embed_dim += 1
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Class embedding (if conditional)
        if conditional:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)

        # Denoising network (predicts noise)
        layers = []
        input_dim = feature_dim + time_embed_dim
        if conditional:
            input_dim += time_embed_dim  # Add class embedding

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, feature_dim))
        self.denoising_net = nn.Sequential(*layers)

        print(f"[FeatureDiffusion] Initialized with:")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Timesteps: {timesteps}")
        print(f"  - Schedule: {beta_schedule}")
        print(f"  - Conditional: {conditional}")
        if conditional:
            print(f"  - Num classes: {num_classes}")

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to features at timestep t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict noise at timestep t (the core denoising function).

        Args:
            x_t: Noisy features at timestep t
            t: Timestep
            y: Class labels (required if conditional=True)
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Combine embeddings
        if self.conditional:
            if y is None:
                raise ValueError("Class labels y must be provided in conditional mode")
            y_emb = self.class_embed(y)
            condition = torch.cat([t_emb, y_emb], dim=1)
        else:
            condition = t_emb

        # Concatenate noisy features with condition
        h = torch.cat([x_t, condition], dim=1)

        # Predict noise
        return self.denoising_net(h)

    def forward(self, x_start: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass: sample random timestep and compute denoising loss.

        Args:
            x_start: Clean features [B, D]
            y: Class labels [B] (required if conditional=True)

        Returns:
            loss: Denoising loss (MSE between predicted and true noise)
            predicted_noise: The predicted noise (for analysis)
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timestep for each sample
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise to features
        x_t = self.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, y)

        # Compute loss (MSE between predicted and true noise)
        loss = F.mse_loss(predicted_noise, noise)

        return loss, predicted_noise

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single denoising step: p(x_{t-1} | x_t)
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = self.predict_noise(x_t, t_tensor, y)

        # Compute mean of p(x_{t-1} | x_t)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean
        else:
            # Add noise
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, y: Optional[torch.Tensor] = None, device: str = "cuda") -> torch.Tensor:
        """
        Full denoising loop: sample from p(x_0) by iteratively denoising from x_T ~ N(0, I)
        """
        batch_size = shape[0]

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t, t, y)

        return x_t

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, num_steps: int = 100) -> torch.Tensor:
        """
        Reconstruct features by adding noise and then denoising.
        Used for computing reconstruction error as anomaly score.

        Args:
            x: Clean features
            y: Class labels (optional, for conditional model)
            num_steps: Number of denoising steps (< timesteps for faster inference)

        Returns:
            Reconstructed features
        """
        device = x.device
        batch_size = x.shape[0]

        # Add noise to a moderate timestep (not full noise)
        t_start = min(num_steps, self.timesteps - 1)
        t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)

        # Denoise back to x_0
        for t in reversed(range(t_start + 1)):
            x_t = self.p_sample(x_t, t, y)

        return x_t

    @torch.no_grad()
    def compute_reconstruction_error(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                                     num_steps: int = 100) -> torch.Tensor:
        """
        Compute reconstruction error as anomaly score for open-set detection.

        Higher reconstruction error indicates the sample is likely from an unknown class.

        Args:
            x: Features to test
            y: Class labels (optional, for conditional model)
            num_steps: Number of denoising steps

        Returns:
            Reconstruction errors [B]
        """
        x_recon = self.reconstruct(x, y, num_steps)
        error = torch.norm(x - x_recon, dim=1)
        return error

    @torch.no_grad()
    def compute_likelihood(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                          num_timesteps: int = 10) -> torch.Tensor:
        """
        Approximate likelihood by computing denoising loss at multiple timesteps.

        Lower likelihood indicates potential open-set sample.

        Args:
            x: Features to evaluate
            y: Class labels (optional)
            num_timesteps: Number of timesteps to sample

        Returns:
            Negative log-likelihood scores [B] (lower is more likely)
        """
        device = x.device
        batch_size = x.shape[0]

        total_loss = 0.0
        timesteps = torch.linspace(0, self.timesteps - 1, num_timesteps, dtype=torch.long, device=device)

        for t_val in timesteps:
            t = t_val.unsqueeze(0).expand(batch_size)
            noise = torch.randn_like(x)
            x_t = self.q_sample(x, t, noise)
            predicted_noise = self.predict_noise(x_t, t, y)
            loss = torch.sum((predicted_noise - noise) ** 2, dim=1)
            total_loss += loss

        return total_loss / num_timesteps


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# =============================================================================
# Multi-Timestep Feature Diffusion (Enhanced Version)
# =============================================================================

class MultiTimestepFeatureDiffusion(nn.Module):
    """
    Enhanced diffusion model that computes anomaly scores using multiple timesteps.

    This provides more robust open-set detection by considering reconstruction quality
    across different noise levels.
    """

    def __init__(self, base_diffusion: FeatureDiffusion):
        super().__init__()
        self.diffusion = base_diffusion

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        return self.diffusion(x, y)

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        method: str = "reconstruction",
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Compute anomaly score for open-set detection.

        Args:
            x: Features to test
            y: Class labels (optional)
            method: "reconstruction" or "likelihood"
            num_steps: Number of denoising steps

        Returns:
            Anomaly scores [B] (higher = more anomalous = likely open-set)
        """
        if method == "reconstruction":
            return self.diffusion.compute_reconstruction_error(x, y, num_steps)
        elif method == "likelihood":
            return self.diffusion.compute_likelihood(x, y, num_steps)
        else:
            raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Factory Function
# =============================================================================

def create_feature_diffusion(
    feature_dim: int,
    num_classes: Optional[int] = None,
    hidden_dims: list = [512, 256, 512],
    timesteps: int = 1000,
    beta_schedule: str = "cosine",
    conditional: bool = True,
    dropout: float = 0.1,
    enhanced: bool = True,
) -> nn.Module:
    """
    Factory function to create feature diffusion model.

    Args:
        feature_dim: Dimension of features
        num_classes: Number of known classes
        hidden_dims: Hidden dimensions for denoising network
        timesteps: Number of diffusion timesteps
        beta_schedule: Beta schedule ("linear" or "cosine")
        conditional: Whether to condition on class labels
        dropout: Dropout rate
        enhanced: Whether to use multi-timestep enhanced version

    Returns:
        Feature diffusion model
    """
    base = FeatureDiffusion(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        conditional=conditional,
        num_classes=num_classes,
        dropout=dropout,
    )

    if enhanced:
        return MultiTimestepFeatureDiffusion(base)
    else:
        return base
