# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Any
import random, numpy as np, torch

# -----------------------------
# 随机种子 & 设备/GPU 解析
# -----------------------------
def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def parse_gpu_ids(gpus_input) -> List[int]:
    if gpus_input is None or gpus_input == "null": return []
    if isinstance(gpus_input, int): return [gpus_input]
    if isinstance(gpus_input, str):
        if gpus_input.lower() in ["null", "none", ""]: return []
        parts = [p.strip() for p in gpus_input.split(',') if p.strip()]
        ids = []
        for p in parts:
            if not p.isdigit(): raise ValueError(f"Invalid GPU id: {p}")
            ids.append(int(p))
        return ids
    if isinstance(gpus_input, (list, tuple)): return [int(x) for x in gpus_input]
    raise ValueError(f"Unsupported GPU input type: {type(gpus_input)}")

def setup_device(which: str = 'auto', gpu_ids: Optional[List[int]] = None) -> torch.device:
    gpu_ids = gpu_ids or []
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            print("CUDA not available, fallback to CPU despite --gpus provided.")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cuda' if (which=='auto' and torch.cuda.is_available()) else which)
    print(f"Using device: {device}")
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            if len(gpu_ids) > 0:
                for i in gpu_ids:
                    name = torch.cuda.get_device_name(i)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    print(f"  cuda:{i} -> {name} | VRAM: {total:.1f} GB")
            else:
                i = device.index or 0
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  cuda:{i} -> {name} | VRAM: {total:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        except Exception:
            pass
    return device

# -----------------------------
# 安全 JSON 序列化转换
# -----------------------------
def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# -----------------------------
# 推理期后处理：Logit Adjustment & τ-norm
# -----------------------------
def logits_logit_adjustment(logits: torch.Tensor, class_counts: np.ndarray, tau: float = 1.0) -> torch.Tensor:
    if tau <= 0:
        return logits
    prior = class_counts.astype(np.float64)
    prior = np.maximum(prior, 1.0)
    prior = prior / prior.sum()
    shift = torch.from_numpy(np.log(prior + 1e-12)).to(logits.device).float()
    return logits - tau * shift

def tau_norm_weights(weight: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    if tau == 0:
        return weight
    norm = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    return weight / (norm ** tau)
