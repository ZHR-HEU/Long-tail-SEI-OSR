# -*- coding: utf-8 -*-
"""
Imbalanced-learning data utilities for 1D signals (I/Q style).

统一目标
--------
1) 纯命令行/关键字参数可用：无需 cfg，默认值合理。
2) 统一返回：Dataset -> (Tensor[C, T], int)，并暴露 dataset.labels / num_classes / class_counts。
3) 易用采样：提供以下采样家族（统一命名 + 文献锚点）：
   - 'inv_freq'         : 逆频率采样（逐样本加权，WeightedRandomSampler 基线）。[经典基线；与 Cui19 作对照]
   - 'class_uniform'    : 类均匀采样（先抽类再类内均匀）。[class-balanced resampler]
   - 'sqrt'             : 平方根频率采样（p(c) ∝ sqrt(n_c)）。[LVIS Repeat-Factor Sampling 的思想]
   - 'power'            : 幂律采样（p(c) ∝ n_c^α）。α∈[0,1]，统一视角：α=0→类均匀；α=0.5→sqrt；α=1→原分布。
   - 'progressive_power': 渐进式幂律采样（α 随 epoch 线性退火），类比 DRW/DRS 的日程化思想。

参考锚点
--------
- Cui et al., CVPR 2019 (Class-balanced Loss)
- NeurIPS 2023 "How Re-sampling Helps for Long-Tail Learning?"（class-balanced resampler）
- LVIS 附录（Repeat-Factor Sampling 思想）
- LDAM-DRW（DRW/DRS 日程）

IO 支持
-------
- .h5/.hdf5（h5py）、.mat（scipy.io.loadmat + v7.3 用 h5py）、.npz/.npy（含配对 _y.npy）。
- 惰性打开，兼容多进程 DataLoader。
- 变长处理：RandomCropOrPad(length)；常用增强：TimeShift/Amplitude/Noise；PerSampleNormalize。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler

# -----------------------------
# 可选依赖（保留原始导入异常，便于报错提示）
# -----------------------------
_h5py_import_error: Optional[BaseException] = None
try:
    import h5py  # type: ignore
except BaseException as e:
    h5py = None  # type: ignore
    _h5py_import_error = e

_loadmat_import_error: Optional[BaseException] = None
try:
    from scipy.io import loadmat  # type: ignore
except BaseException as e:
    loadmat = None  # type: ignore
    _loadmat_import_error = e


# -----------------------------
# 小型变换库（无需 torchvision）
# -----------------------------
class Compose:
    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.from_numpy(x)
        return t.float()


class RandomTimeShift:
    def __init__(self, max_shift: int = 32, p: float = 0.5):
        self.max_shift = int(max_shift)
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0 or random.random() > self.p:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class RandomAmplitude:
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        bias_range: Tuple[float, float] = (-0.05, 0.05),
        p: float = 0.5,
    ):
        self.scale_range = scale_range
        self.bias_range = bias_range
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        s = random.uniform(*self.scale_range)
        b = random.uniform(*self.bias_range)
        return x * s + b


class RandomGaussianNoise:
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.02), p: float = 0.5):
        self.std_range = std_range
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        std = random.uniform(*self.std_range)
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std


class RandomCropOrPad:
    """将时序修剪/补零到固定长度 length。输入 [C, T]。"""

    def __init__(self, length: int):
        self.length = int(length)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, T = x.shape[-2], x.shape[-1]
        if T == self.length:
            return x
        if T > self.length:
            start = random.randint(0, T - self.length)
            return x[..., start : start + self.length]
        else:
            pad_len = self.length - T
            pad = torch.zeros(C, pad_len, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-1)


class PerSampleNormalize:
    """每条样本按通道零均值单位方差。"""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std


# -----------------
# 辅助函数（MAT 读取）
# -----------------
def _require_h5py_for_v73():
    if h5py is None:
        hint = f"\n原始导入异常: {_h5py_import_error!r}" if _h5py_import_error else ""
        raise ImportError(
            "检测到 MATLAB v7.3 / HDF5 文件，但当前环境无法使用 h5py。\n"
            "请在**运行训练的同一 Python 环境**中安装或修复 h5py：\n"
            "  conda install -n zhahaoranEnv h5py    # 或\n"
            "  pip install --upgrade --force-reinstall h5py\n"
            "若还有二进制依赖问题，可尝试：\n"
            "  conda install -n zhahaoranEnv hdf5\n"
            "  conda install -n zhahaoranEnv 'numpy<2'\n"
            + hint
        )


def _require_scipy_for_legacy_mat():
    if loadmat is None:
        hint = f"\n原始导入异常: {_loadmat_import_error!r}" if _loadmat_import_error else ""
        raise ImportError(
            "需要 SciPy 才能读取非 v7.3 的 MATLAB .mat 文件：\n"
            "  conda install -n zhahaoranEnv scipy    # 或 pip install scipy" + hint
        )


def _is_mat_v73(path: str) -> bool:
    """
    以“能否用 h5py 打开”为准：能打开即视为 v7.3(HDF5)。
    这样和参考实现一致，避免仅凭魔数造成误判。
    """
    if h5py is None:
        return False
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def _load_mat_any(path: str) -> Dict[str, np.ndarray]:
    """
    统一入口：v7.3 -> h5py；非 v7.3 -> scipy.io.loadmat。
    键名候选与维度整理对齐参考实现：
      - 数据键候选：["X", "data", "signal", "signals"]
      - 标签键候选：["Y", "label", "labels", "y"]
      - 若 X 为 [C, T, N] 且 N == len(Y) 则转置为 [N, C, T]
    """
    data_keys = ["X", "data", "signal", "signals"]
    label_keys = ["Y", "label", "labels", "y"]

    if _is_mat_v73(path):
        _require_h5py_for_v73()
        with h5py.File(path, "r") as f:
            key_x = next((k for k in data_keys if k in f), None)
            key_y = next((k for k in label_keys if k in f), None)
            if key_x is None or key_y is None:
                raise KeyError(f"Cannot find data/label keys in {path}. Found keys: {list(f.keys())}")
            X = np.array(f[key_x])
            Y = np.array(f[key_y]).squeeze()
            # [C, T, N] -> [N, C, T]
            if X.ndim == 3 and X.shape[-1] == len(Y):
                X = np.transpose(X, (2, 0, 1))
            return {"X": X, "Y": Y}

    # legacy mat via SciPy
    _require_scipy_for_legacy_mat()
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        # SciPy 报 v7.3，但我们没能提前判出 → 回退到 h5py
        _require_h5py_for_v73()
        with h5py.File(path, "r") as f:
            key_x = next((k for k in data_keys if k in f), None)
            key_y = next((k for k in label_keys if k in f), None)
            if key_x is None or key_y is None:
                raise KeyError(f"Cannot find data/label keys in {path}. Found keys: {list(f.keys())}") from e
            X = np.array(f[key_x])
            Y = np.array(f[key_y]).squeeze()
            if X.ndim == 3 and X.shape[-1] == len(Y):
                X = np.transpose(X, (2, 0, 1))
            return {"X": X, "Y": Y}

    key_x = next((k for k in data_keys if k in mat), None)
    key_y = next((k for k in label_keys if k in mat), None)
    if key_x is None or key_y is None:
        raise KeyError(f"Cannot find data/label keys in {path}. Found keys: {list(mat.keys())}")
    X = np.asarray(mat[key_x])
    Y = np.asarray(mat[key_y]).squeeze()
    if X.ndim == 3 and X.shape[-1] == len(Y):
        X = np.transpose(X, (2, 0, 1))
    return {"X": X, "Y": Y}


def compute_class_counts(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    C = int(num_classes) if num_classes is not None else (int(labels.max()) + 1 if labels.size else 0)
    counts = np.bincount(labels, minlength=C)
    return counts.astype(np.int64)


def make_long_tailed_indices(
        labels: np.ndarray, num_classes: int, imbalance_ratio: float, seed: int = 42
) -> np.ndarray:
    """
    Create long-tailed indices from labels with exponential decay.

    Args:
        labels: Array of class labels
        num_classes: Total number of classes
        imbalance_ratio: Ratio between max and min class (e.g., 100.0)
        seed: Random seed

    Returns:
        Array of selected indices creating long-tail distribution
    """
    np.random.seed(seed)
    labels = np.asarray(labels).astype(int)

    print(f"Input labels shape: {labels.shape}")
    print(f"Unique classes in labels: {np.unique(labels)}")
    print(f"Target imbalance ratio: {imbalance_ratio}")

    # 计算每个类别的原始样本数
    original_counts = compute_class_counts(labels, num_classes)
    print(f"Original class counts: {original_counts.tolist()}")

    # 找到最大的原始样本数作为头部类的目标数量
    max_original = original_counts.max()
    print(f"Max original count: {max_original}")

    # 计算每个类的目标样本数，使用指数衰减
    target_counts = np.zeros(num_classes, dtype=int)

    for i in range(num_classes):
        # 指数衰减：第i个类的样本数 = max_count * (1/ratio)^(i/(C-1))
        if num_classes == 1:
            decay_factor = 1.0
        else:
            decay_factor = (1.0 / imbalance_ratio) ** (i / (num_classes - 1))

        target_count = int(max_original * decay_factor)
        # 确保每个类至少有1个样本，但不超过原有样本数
        target_count = max(1, min(target_count, original_counts[i]))
        target_counts[i] = target_count

    print(f"Target counts: {target_counts.tolist()}")

    # 验证目标比例
    actual_target_ratio = target_counts.max() / max(1, target_counts.min())
    print(f"Target ratio will be: {actual_target_ratio:.2f}")

    # 选择样本
    selected_indices = []

    for class_id in range(num_classes):
        # 找到该类的所有样本索引
        class_mask = (labels == class_id)
        class_indices = np.where(class_mask)[0]

        target_count = target_counts[class_id]
        available_count = len(class_indices)

        # print(f"Class {class_id}: need {target_count}, available {available_count}")

        if target_count <= 0 or available_count <= 0:
            print(f"  -> Skipping class {class_id} (no samples needed or available)")
            continue

        if target_count >= available_count:
            # 需要的样本数大于等于可用数，全部选择
            selected = class_indices
            # print(f"  -> Selected all {len(selected)} samples")
        else:
            # 随机选择目标数量的样本
            selected = np.random.choice(class_indices, size=target_count, replace=False)
            # print(f"  -> Randomly selected {len(selected)} samples")

        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices, dtype=int)
    print(f"Total selected indices: {len(selected_indices)}")

    # 验证结果
    if len(selected_indices) > 0:
        selected_labels = labels[selected_indices]
        final_counts = compute_class_counts(selected_labels, num_classes)
        final_ratio = final_counts.max() / max(1, final_counts.min()) if final_counts.min() > 0 else float('inf')

        print(f"Final verification:")
        print(f"  Final class counts: {final_counts.tolist()}")
        print(f"  Final imbalance ratio: {final_ratio:.2f}")
        print(f"  Target was: {imbalance_ratio:.2f}")

        if abs(final_ratio - imbalance_ratio) > imbalance_ratio * 0.2:
            print(f"  WARNING: Final ratio {final_ratio:.2f} differs significantly from target {imbalance_ratio:.2f}")
    else:
        print("ERROR: No samples selected!")
        return np.array([], dtype=int)

    return selected_indices


# -----------------
# 数据集
# -----------------
class ADSBSignalDataset(Dataset):
    """
    统一读取 I/Q 或多通道时序信号，输出 (Tensor[C, T], int)。

    形状约定
    -------
    支持 [N, C, T] 或 [N, T, C]；单样本内部一律整理为 [C, T]。

    数据键约定
    ---------
    候选键：X/data/signal/signals 与 Y/label/labels/y。

    属性
    ----
    - labels: np.ndarray[int]
    - num_classes: int
    - class_counts: np.ndarray[int]
    """
    CAND_KEYS_X = ("signals", "signal", "X", "data", "x")
    CAND_KEYS_Y = ("labels", "label", "y", "Y", "target", "targets")

    def __init__(
        self,
        path: str,
        split: Optional[str] = None,
        data_key: Optional[str] = None,
        label_key: Optional[str] = None,
        indices: Optional[Sequence[int]] = None,
        in_memory: bool = False,
        target_length: Optional[int] = None,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        seed: int = 2027,
    ):
        super().__init__()
        self.path = os.path.abspath(path)
        self.split = split
        self.data_key = data_key
        self.label_key = label_key
        self.in_memory = bool(in_memory)
        self.target_length = target_length
        self.transforms = transforms
        self.normalize = bool(normalize)
        self.seed = int(seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._h5: Any = None  # 懒打开

        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".npy":
            arr = np.load(self.path, allow_pickle=False)
            if self.split is not None and isinstance(arr, dict) and self.split in arr:
                arr = arr[self.split]
            assert isinstance(arr, np.ndarray)
            y_path = self.path.replace(".npy", "_y.npy")
            if not os.path.exists(y_path):
                raise FileNotFoundError(f"labels file not found: {y_path}")
            labels = np.load(y_path, allow_pickle=False)
            self._X, self._Y = arr, labels

        elif ext == ".npz":
            data = np.load(self.path, allow_pickle=False)
            kx = self.data_key or (self.split or self._guess_key(tuple(data.keys()), ADSBSignalDataset.CAND_KEYS_X))
            ky = self.label_key or self._guess_key(tuple(data.keys()), ADSBSignalDataset.CAND_KEYS_Y)
            self._X = data[kx]
            self._Y = data[ky]

        elif ext == ".mat":
            mat = _load_mat_any(self.path)
            kx = self.data_key or (self.split or self._guess_key(tuple(mat.keys()), ADSBSignalDataset.CAND_KEYS_X))
            ky = self.label_key or self._guess_key(tuple(mat.keys()), ADSBSignalDataset.CAND_KEYS_Y)
            self._X = np.asarray(mat[kx])
            self._Y = np.asarray(mat[ky]).reshape(-1)

        elif ext in (".h5", ".hdf5"):
            _require_h5py_for_v73()

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # 统一索引映射
        if self._Y is not None:
            labels = np.asarray(self._Y).astype(np.int64).reshape(-1)
            N = int(labels.shape[0])
        else:
            N = self._resolve_length_h5()
            labels = self._read_labels_h5(slice(0, N))

        if indices is None:
            self.indices = np.arange(N, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)

        self.labels = labels[self.indices]
        self.num_classes = int(self.labels.max()) + 1 if self.labels.size else 0
        self.class_counts = compute_class_counts(self.labels, self.num_classes)

        # 变换链
        chain: List[Callable[[torch.Tensor], torch.Tensor]] = [ToTensor()]
        if self.target_length is not None:
            chain.append(RandomCropOrPad(self.target_length))
        if self.normalize:
            chain.append(PerSampleNormalize())
        if self.transforms is not None:
            chain.append(self.transforms)
        self.pipeline = Compose(chain)

    # ----- HDF5 helpers -----
    def _open_h5(self):
        if self._h5 is None:
            _require_h5py_for_v73()
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def _guess_key(self, keys: Sequence[str], candidates: Sequence[str]) -> str:
        keys_l = [k.lower() for k in keys]
        for c in candidates:
            if c in keys_l:
                return keys[keys_l.index(c)]
        return keys[0]

    def _resolve_length_h5(self) -> int:
        f = self._open_h5()
        kx = self.data_key or (self.split or self._guess_key(list(f.keys()), ADSBSignalDataset.CAND_KEYS_X))
        return int(f[kx].shape[0])

    def _read_labels_h5(self, sl: slice | np.ndarray) -> np.ndarray:
        f = self._open_h5()
        ky = self.label_key or self._guess_key(list(f.keys()), ADSBSignalDataset.CAND_KEYS_Y)
        y = np.asarray(f[ky][sl]).reshape(-1)
        return y.astype(np.int64)

    def _read_item_h5(self, idx: int) -> Tuple[np.ndarray, int]:
        f = self._open_h5()
        kx = self.data_key or (self.split or self._guess_key(list(f.keys()), ADSBSignalDataset.CAND_KEYS_X))
        ky = self.label_key or self._guess_key(list(f.keys()), ADSBSignalDataset.CAND_KEYS_Y)
        x = np.asarray(f[kx][idx])
        y = int(np.asarray(f[ky][idx]).reshape(()))
        return x, y

    # ----- dataset core -----
    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _ensure_CT(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 2:
            pass
        elif arr.ndim == 3:
            # 兼容 [C, T] 或 [T, C] 在第一维多出的情况（少见）
            # 策略：较小的维度当通道，较大的当时间
            if arr.shape[0] in (1, 2, 3, 4) and arr.shape[1] > arr.shape[0]:
                arr = arr[0:arr.shape[0], :, :].mean(axis=0) if arr.shape[0] > 1 else arr[0]
            elif arr.shape[1] in (1, 2, 3, 4) and arr.shape[2] > arr.shape[1]:
                arr = arr[:, :, :]
                arr = np.transpose(arr, (1, 0, 2)).mean(axis=0) if arr.shape[0] > 1 else arr[0]
            else:
                arr = np.moveaxis(arr, -1, 0)
        else:
            # >=4 维：将最后一维视为时间、倒数第二维视为通道
            arr = np.moveaxis(arr, (-2, -1), (0, 1))
        # 最终整理为 [C, T]
        if arr.ndim != 2:
            if arr.shape[0] > arr.shape[-1] and arr.shape[-1] in (1, 2, 3, 4):
                arr = np.transpose(arr, (1, 0))
        if arr.shape[0] not in (1, 2, 3, 4) and arr.shape[1] in (1, 2, 3, 4):
            arr = np.transpose(arr, (1, 0))
        return arr

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        if self._X is not None:
            x = self._X[idx]
            y = int(self._Y[idx])
        else:
            x, y = self._read_item_h5(idx)
        x = self._ensure_CT(x)
        x_t = self.pipeline(x)
        return x_t, y


# -----------------------------
# Sampler 与工厂（统一命名 + 向后兼容）
# -----------------------------
class _ClassBuckets:
    """把索引按类别分桶，提供按"先抽类、再抽样本"的访问。"""

    def __init__(self, labels: np.ndarray, seed: int = 0):
        self.labels = np.asarray(labels).astype(np.int64)
        self.num_classes = int(self.labels.max()) + 1 if self.labels.size else 0
        self.class_to_indices: List[np.ndarray] = []
        rng = np.random.default_rng(seed)
        for c in range(self.num_classes):
            idx = np.where(self.labels == c)[0]
            if idx.size:
                rng.shuffle(idx)
            self.class_to_indices.append(idx)
        self.rng = rng

    def sample_from_class(self, c: int) -> int:
        arr = self.class_to_indices[c]
        if arr.size == 0:
            non_empty = [i for i, a in enumerate(self.class_to_indices) if a.size > 0]
            c = int(self.rng.choice(non_empty))
            arr = self.class_to_indices[c]
        j = int(self.rng.integers(0, arr.size))
        return int(arr[j])


class ClassUniformSampler(Sampler[int]):
    """类均匀采样（Class-balanced resampler）：先均匀抽类，再类内均匀（有放回）。"""

    def __init__(self, labels: np.ndarray, num_samples: Optional[int] = None, seed: int = 0):
        self.buckets = _ClassBuckets(labels, seed=seed)
        self.num_samples = int(num_samples) if num_samples is not None else len(labels)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        C = max(1, self.buckets.num_classes)
        for _ in range(self.num_samples):
            c = int(self.buckets.rng.integers(0, C))
            yield self.buckets.sample_from_class(c)


class PowerLawSampler(Sampler[int]):
    """幂律采样：p(c) ∝ n_c^alpha；alpha=0→类均匀；alpha=0.5→sqrt；alpha=1→原分布。"""

    def __init__(self, labels: np.ndarray, alpha: float = 0.5, num_samples: Optional[int] = None, seed: int = 0):
        self.labels = np.asarray(labels).astype(np.int64)
        self.buckets = _ClassBuckets(self.labels, seed=seed)
        self.alpha = float(alpha)
        self.num_samples = int(num_samples) if num_samples is not None else len(labels)
        self._recompute_probs()

    def _recompute_probs(self):
        counts = compute_class_counts(self.labels, self.buckets.num_classes).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        probs = counts ** float(self.alpha)
        self.probs = (probs / probs.sum()).astype(np.float64)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        C = self.buckets.num_classes
        classes = np.arange(C)
        for _ in range(self.num_samples):
            c = int(self.buckets.rng.choice(classes, p=self.probs))
            yield self.buckets.sample_from_class(c)


class ProgressivePowerSampler(Sampler[int]):
    """渐进式幂律采样：α 从 alpha_start → alpha_end（线性），p(c) ∝ n_c^α。"""

    def __init__(
        self,
        labels: np.ndarray,
        alpha_start: float = 0.5,
        alpha_end: float = 0.0,
        total_epochs: int = 100,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        self.labels = np.asarray(labels).astype(np.int64)
        self.buckets = _ClassBuckets(self.labels, seed=seed)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.total_epochs = max(1, int(total_epochs))
        self.current_epoch = 0
        self.num_samples = int(num_samples) if num_samples is not None else len(labels)
        self.current_alpha = self.alpha_start
        self._recompute_probs()

    def set_epoch(self, epoch: int):
        self.current_epoch = max(0, int(epoch))
        t = min(self.current_epoch / self.total_epochs, 1.0)
        self.current_alpha = (1 - t) * self.alpha_start + t * self.alpha_end
        self._recompute_probs()

    def _recompute_probs(self):
        counts = compute_class_counts(self.labels, self.buckets.num_classes).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        probs = counts ** float(self.current_alpha)
        self.probs = (probs / probs.sum()).astype(np.float64)

    def get_progress_info(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_alpha": self.current_alpha,
            "progress": self.current_epoch / self.total_epochs,
            "class_probs": self.probs.tolist(),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        C = self.buckets.num_classes
        classes = np.arange(C)
        for _ in range(self.num_samples):
            c = int(self.buckets.rng.choice(classes, p=self.probs))
            yield self.buckets.sample_from_class(c)


# -----------------------------
# 采样器工厂（统一命名 + 向后兼容）
# -----------------------------
_SAMPLER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "none": ("none", "natural", "instance", None),
    "inv_freq": ("inv_freq", "balanced", "weighted"),
    "class_uniform": ("class_uniform", "balance", "class-balanced"),
    "sqrt": ("sqrt", "square", "square_root"),
    "power": ("power",),
    "progressive_power": ("progressive_power", "progressive"),
}


def _canonize_sampler_name(name: Optional[str]) -> str:
    if name is None:
        return "none"
    key = str(name).lower()
    for canonical, aliases in _SAMPLER_ALIASES.items():
        if key in [a for a in aliases if a is not None]:
            return canonical
    if key in _SAMPLER_ALIASES:
        return key
    raise ValueError(f"Unsupported sampler name: {name}. Valid: {list(_SAMPLER_ALIASES.keys())}")


def make_sampler(
    labels: np.ndarray,
    method: Optional[str],
    *,
    seed: int = 0,
    alpha: float = 0.5,
    alpha_start: float = 0.5,
    alpha_end: float = 0.0,
    total_epochs: int = 100,
) -> Optional[Sampler[int]]:
    """
    统一创建采样器（按规范名）：
    - 'none'               : 不改分布，DataLoader 用 shuffle=True。
    - 'inv_freq'           : 逆频率采样（WeightedRandomSampler）。
    - 'class_uniform'      : 类均匀采样（先类后样本）。
    - 'sqrt'               : 平方根频率（等价 power(alpha=0.5)）。
    - 'power'              : 幂律采样（alpha 可调）。
    - 'progressive_power'  : 渐进式幂律（alpha_start→alpha_end）。
    """
    m = _canonize_sampler_name(method)
    labels = np.asarray(labels).astype(np.int64)
    if m == "none" or labels.size == 0:
        return None

    counts = compute_class_counts(labels)
    if counts.size == 0:
        return None

    if m == "inv_freq":
        w_per_class = 1.0 / np.clip(counts, 1, None)
        w = w_per_class[labels]
        return WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
        )

    if m == "class_uniform":
        return ClassUniformSampler(labels, num_samples=len(labels), seed=seed)

    if m == "sqrt":
        return PowerLawSampler(labels, alpha=0.5, num_samples=len(labels), seed=seed)

    if m == "power":
        return PowerLawSampler(labels, alpha=float(alpha), num_samples=len(labels), seed=seed)

    if m == "progressive_power":
        return ProgressivePowerSampler(
            labels,
            alpha_start=float(alpha_start),
            alpha_end=float(alpha_end),
            total_epochs=int(total_epochs),
            num_samples=len(labels),
            seed=seed,
        )
    raise ValueError(f"Unsupported sampler (canonical): {m}")


def update_sampler_epoch(sampler: Optional[Sampler[int]], epoch: int) -> bool:
    """若采样器支持 epoch 更新则进行（progressive_power）。"""
    if sampler is None:
        return False
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))
        return True
    return False


def get_sampler_info(sampler: Optional[Sampler[int]]) -> Optional[Dict[str, Any]]:
    """返回当前采样概率等信息（若可用）。"""
    if sampler is None:
        return None
    if hasattr(sampler, "get_progress_info"):
        return sampler.get_progress_info()
    if hasattr(sampler, "probs"):
        return {"class_probs": getattr(sampler, "probs").tolist(), "alpha": getattr(sampler, "alpha", None)}
    return None


# -----------------------------
# DataLoader 构建
# -----------------------------
@dataclass
class LoaderConfig:
    path_train: str
    path_val: Optional[str] = None
    path_test: Optional[str] = None

    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # dataset params
    target_length: Optional[int] = None
    normalize: bool = True
    in_memory: bool = False

    # 采样方法（规范名或别名）
    sampler: Optional[str] = None
    # power 系列参数
    sampler_alpha: float = 0.5
    # progressive_power 参数
    sampler_alpha_start: float = 0.5
    sampler_alpha_end: float = 0.0
    sampler_total_epochs: int = 100

    seed: int = 2027


def _build_transforms(cfg: LoaderConfig, for_train: bool) -> Compose:
    ts: List[Callable[[torch.Tensor], torch.Tensor]] = []
    if for_train:
        ts.append(RandomTimeShift(max_shift=32, p=0.5))
        ts.append(RandomAmplitude())
        ts.append(RandomGaussianNoise())
    return Compose(ts) if ts else Compose([])


def build_dataloaders(cfg: LoaderConfig) -> Dict[str, Any]:
    """构建 train/val/test 的 Dataset 与 DataLoader；返回字典含采样器句柄。"""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ds_tr = ADSBSignalDataset(
        path=cfg.path_train,
        target_length=cfg.target_length,
        transforms=_build_transforms(cfg, for_train=True),
        normalize=cfg.normalize,
        in_memory=cfg.in_memory,
        seed=cfg.seed,
    )

    sampler = make_sampler(
        ds_tr.labels,
        cfg.sampler,
        seed=cfg.seed,
        alpha=cfg.sampler_alpha,
        alpha_start=cfg.sampler_alpha_start,
        alpha_end=cfg.sampler_alpha_end,
        total_epochs=cfg.sampler_total_epochs,
    )

    train_loader = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=(cfg.num_workers > 0),
    )

    def _mk_eval_loader(path_opt: Optional[str]) -> Tuple[Optional[ADSBSignalDataset], Optional[DataLoader]]:
        if path_opt is None:
            return None, None
        ds = ADSBSignalDataset(
            path=path_opt,
            target_length=cfg.target_length,
            transforms=_build_transforms(cfg, for_train=False),
            normalize=cfg.normalize,
            in_memory=cfg.in_memory,
            seed=cfg.seed,
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            persistent_workers=(cfg.num_workers > 0),
        )
        return ds, loader

    ds_va, val_loader = _mk_eval_loader(cfg.path_val)
    ds_te, test_loader = _mk_eval_loader(cfg.path_test)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": ds_tr,
        "val_dataset": ds_va,
        "test_dataset": ds_te,
        "num_classes": ds_tr.num_classes,
        "class_counts": ds_tr.class_counts,
        "sampler": sampler,  # 返回采样器便于 epoch 管理/监控
    }


# -----------------------------
# 包装器（progressive_power 便捷用法）
# -----------------------------
class ProgressiveDataLoader:
    """包装器：自动处理渐进式采样的 epoch 更新。"""

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg
        self.loaders = build_dataloaders(cfg)
        self.train_loader = self.loaders["train"]
        self.sampler = self.loaders["sampler"]
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        if update_sampler_epoch(self.sampler, self.current_epoch):
            info = get_sampler_info(self.sampler)
            if info and "current_alpha" in info:
                print(
                    f"[Progressive] epoch={self.current_epoch} alpha={info['current_alpha']:.3f} "
                    f"progress={info['progress']:.1%}"
                )

    def get_loader(self, split: str):
        return self.loaders[split]

    def get_train_loader(self):
        return self.train_loader

    def get_sampler_info(self) -> Optional[Dict[str, Any]]:
        return get_sampler_info(self.sampler)

    @property
    def num_classes(self):
        return self.loaders["num_classes"]

    @property
    def class_counts(self):
        return self.loaders["class_counts"]

    def print_class_distribution(self):
        counts = self.class_counts
        total = int(counts.sum())
        print(f"Class distribution (N={total}):")
        for i, cnt in enumerate(counts):
            pct = 100.0 * (cnt / max(1, total))
            print(f"  Class {i}: {cnt} ({pct:.1f}%)")
        info = self.get_sampler_info()
        if info and "class_probs" in info:
            print("Current sampling probs:")
            for i, p in enumerate(info["class_probs"]):
                print(f"  Class {i}: {p:.3f}")


# -----------------------------
# CLI 入口（可选）
# -----------------------------
def dataclass_fields(dc) -> List[Any]:
    return list(getattr(dc, "__dataclass_fields__").values())


def build_from_cli(**kwargs) -> Dict[str, Any]:
    valid = {f.name for f in dataclass_fields(LoaderConfig)}
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid}
    cfg = LoaderConfig(**cfg_kwargs)  # type: ignore[arg-type]
    return build_dataloaders(cfg)


__all__ = [
    "ADSBSignalDataset",
    "LoaderConfig",
    "build_dataloaders",
    "make_sampler",
    "update_sampler_epoch",
    "get_sampler_info",
    "ProgressiveDataLoader",
    # samplers
    "ClassUniformSampler",
    "PowerLawSampler",
    "ProgressivePowerSampler",
]
