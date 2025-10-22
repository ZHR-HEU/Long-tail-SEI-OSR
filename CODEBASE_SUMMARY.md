# Long-tail SEI-OSR Codebase Analysis

## Executive Summary

This is a **complete, modular deep learning framework for imbalanced classification** focusing on long-tail learning for signal processing (ADS-B signal recognition). The codebase is well-organized with ~7,500 lines of Python code implementing 15+ imbalanced learning techniques that can be extended for open-set recognition with diffusion models.

**Key Stats:**
- 13 Python modules
- ~7,500 lines of code
- Supports multi-GPU training with AMP (Automatic Mixed Precision)
- Complete pipeline: data loading → training → evaluation → visualization
- Configuration-driven via Hydra/OmegaConf

---

## 1. PROJECT STRUCTURE

```
Long-tail-SEI-OSR/
├── config.yaml                 # Main configuration file (Hydra)
├── main.py                     # Entry point - complete training pipeline
│
├── Data & Sampling
├── data_utils.py              # Dataset loaders, samplers, augmentations
│
├── Model Architecture
├── models.py                  # 15+ neural network architectures
│
├── Loss Functions
├── imbalanced_losses.py       # 10+ loss functions for imbalanced learning
│
├── Training Infrastructure
├── train_eval.py              # Training/validation/test loops
├── stage2.py                  # Two-stage training (CRT, fine-tune)
├── optim_utils.py             # Optimizers, schedulers, loss builders
├── training_utils.py          # Warmup schedulers, checkpointing
│
├── Analysis & Visualization
├── analysis.py                # Classification metrics, per-class analysis
├── visualization.py           # Publication-quality plots
├── summarize.py               # Batch experiment comparison
│
├── Utilities
├── common.py                  # Device setup, logit adjustment, tau-norm
├── trainer_logging.py         # Training logger
├── README.md                  # Comprehensive documentation
└── .git/                      # Git repository
```

---

## 2. CORE DATA LOADING MODULES (data_utils.py)

### Architecture
**File Size:** 36 KB | **Key Classes:** 15+ dataset/sampler classes

### Main Components

#### 1. **Dataset Class**
```python
ADSBSignalDataset(
    path: str,
    target_length: int = 4800,
    normalize: bool = True,
    in_memory: bool = True,
    indices: Optional[np.ndarray] = None
)
```
- Supports multiple formats: `.mat` (scipy/h5py), `.h5`, `.hdf5`, `.npy`, `.npz`
- Lazy loading or in-memory caching
- Handles variable-length sequences (padding/cropping)
- Exposes: `dataset.labels`, `dataset.class_counts`, `dataset.num_classes`

#### 2. **Sampling Strategies** (Unified Factory)
```python
make_sampler(
    labels: np.ndarray,
    method: str,  # 'none', 'inv_freq', 'class_uniform', 'sqrt', 'power', 'progressive_power'
    alpha: float = 0.5,
    alpha_start: float = 0.5,
    alpha_end: float = 0.0,
    total_epochs: int = 100,
    seed: int = 42
)
```

**Implemented Methods:**
- `'none'`: Original distribution (natural sampling)
- `'inv_freq'`: Inverse frequency (p∝1/n_c)
- `'class_uniform'`: Uniform per-class (sample class first, then within-class uniform)
- `'sqrt'`: Square-root frequency (p∝√n_c)
- `'power'`: Power-law sampling (p∝n_c^α, α∈[0,1])
- `'progressive_power'`: Progressive power sampling (α decays during training)

**Design Principles:**
- α=0 → class-uniform
- α=0.5 → sqrt resampling  
- α=1 → original distribution
- Progressive version: Linear decay of α from start→end

#### 3. **Data Augmentation Pipeline**
```python
Compose([
    RandomTimeShift(max_shift=32, p=0.5),      # Temporal translation
    RandomAmplitude(scale_range=(0.9, 1.1), p=0.5),  # Amplitude scaling
    RandomGaussianNoise(std_range=(0.0, 0.02), p=0.5),  # Noise injection
    RandomCropOrPad(target_length),            # Length normalization
    PerSampleNormalize(eps=1e-6),              # Per-sample standardization
    ToTensor()
])
```

#### 4. **Long-tail Creation Utility**
```python
make_long_tailed_indices(
    labels: np.ndarray,
    num_classes: int,
    imbalance_ratio: float = 100.0,  # max_class / min_class
    seed: int = 42
)
```
Creates exponential decay from head→tail classes

#### 5. **DataLoader Factory**
```python
build_dataloaders(
    cfg: LoaderConfig,
    # Returns: {'train', 'val', 'test', 'train_dataset', 'num_classes', 'class_counts'}
)
```

### Key Features
- **Multi-format Support**: Automatically detects and loads .mat, .h5, .npy files
- **Memory Efficient**: Lazy loading or full in-memory options
- **Flexible Sampling**: All samplers inherit from PyTorch's `Sampler` base class
- **Reproducible**: Seed control for all random operations

---

## 3. MODEL ARCHITECTURES (models.py)

### Architecture
**File Size:** 41 KB | **Model Variants:** 15+

### 1D Signal Processing Models

#### **ConvNetADSB** (Default)
```python
ConvNetADSB(
    num_classes: int,
    in_channels: int = 2,       # I/Q samples
    dropout_rate: float = 0.1,
    use_attention: bool = True, # Channel attention
    norm_kind: str = 'auto'     # auto/bn/gn/ln
)
```
- 8-layer deep convolution blocks
- ~350 channels
- Optional channel-wise attention (SE-Net style)
- Flexible normalization (BatchNorm/GroupNorm/LayerNorm)

#### **ResNet1D** (Multiple Depths)
```python
ResNet1D(
    depth: int,  # 2, 3, 4, or 5
    num_classes: int,
    dropout_rate: float = 0.1
)
```
- 1D residual blocks with skip connections
- Progressive depth options

#### **DilatedTCN** (Temporal Convolutional Network)
- Dilated causal convolutions
- Captures long-range temporal dependencies

#### **Frequency-Domain Experts**
```python
FrequencyDomainExpert()
ResNetFrequencyExpert()
```
- FFT-based feature extraction (magnitude, phase, log-power)
- Signal processing specialized

#### **Mixture of Experts (MoE)**
```python
MixtureOfExpertsConvNet()  # Time-domain + Frequency-domain + TCN fusion
MixtureOfExpertsResNet()
```
- Multiple expert branches (ConvNet, ResNet, Frequency-Domain)
- Learnable routing/gating
- Load-balance regularizer to prevent expert collapse

### Classification Heads

#### **Standard Linear**
```python
nn.Linear(in_features, num_classes)
```

#### **CosineMarginClassifier** (LDAM-Ready)
```python
CosineMarginClassifier(
    in_features, num_classes,
    scale: float = 30.0,
    margins: Optional[np.ndarray] = None  # Per-class LDAM margins
)
```
- Normalized feature & weight (cosine similarity)
- Per-class margin subtraction
- Used in LDAM-DRW methods

#### **LogitAdjustedLinear** (Menon et al.)
```python
LogitAdjustedLinear(
    in_features, num_classes,
    class_counts: np.ndarray,
    tau: float = 1.0
)
```
- Built-in logit adjustment: logits - τ*log(π_c)
- Calibrated prior initialization

#### **TemperatureScaledClassifier**
```python
TemperatureScaledClassifier(
    in_features, num_classes,
    initial_temperature: float = 1.5
)
```
- Learnable temperature scaling
- Post-hoc calibration

### Initialization Utilities

```python
class EnhancedClassifierInitializer:
    - balanced_xavier_init()          # Xavier + log-prior bias
    - he_kaiming_init()               # Kaiming + log-prior bias
    - set_bias_to_log_prior()         # Logit Adjustment initialization
    - frequency_aware_bias_init()     # Class-frequency aware bias
```

### Helper Functions

```python
create_model(name: str, num_classes: int, **kwargs)
init_model_for_imbalanced(model, class_counts, init_type, use_log_prior_bias, temperature)
swap_classifier(model, head: str, class_counts, **head_kwargs)
ldam_margins_from_counts(class_counts, power=0.25, max_m=0.5)
```

### Design Features
- **Normalization Factory**: Auto-detect optimal normalization (BN→GN→LN)
- **Flexible Initialization**: Multiple strategies for imbalanced scenarios
- **Modular Heads**: Easy swapping of classification heads
- **Signal-Specialized**: 1D-optimized architectures (not just 1D versions of 2D models)

---

## 4. LOSS FUNCTIONS (imbalanced_losses.py)

### Architecture
**File Size:** 50 KB | **Loss Functions:** 10+

### Loss Function Categories

#### **A. Basic Losses**
```python
CrossEntropyLoss(label_smoothing=0.0)
FocalLoss(gamma=2.0, alpha=0.25)
```

#### **B. Re-weighting Losses**
```python
ClassBalancedLoss(beta=0.9999)      # Cui et al., CVPR 2019
LDAMLoss(
    max_margin=0.5,
    scale=30.0,
    drw_start_epoch=160             # Deferred Re-Weighting
)
ProgressiveLoss(
    total_epochs=200,
    start_strategy='uniform',
    end_strategy='inverse'
)
```

#### **C. Post-hoc Adjustment Losses**
```python
BalancedSoftmaxLoss()               # Adjust posterior π_c'
LogitAdjustmentLoss(tau=1.0)        # Add τ*log(π_c) adjustment
```

#### **D. Cost-Sensitive Losses** (Auto-Cost)
```python
CostSensitiveCE(
    cost_strategy: str  # 'auto'/'sqrt'/'log'/'uniform'/'manual'
)
CostSensitiveExpected()             # Bayesian Risk Minimization
CostSensitiveFocal(gamma=2.0)
```

### Utility Classes

#### **WeightComputer**
```python
WeightComputer.class_priors(class_counts)
WeightComputer.inverse_freq_weights(class_counts)
WeightComputer.sqrt_freq_weights(class_counts)
WeightComputer.log_freq_weights(class_counts)
WeightComputer.effective_num_weights(class_counts, beta)
```

#### **CostMatrix Generator**
```python
# Auto-generates cost matrices from class counts
# Strategies: inv_freq, sqrt, log, uniform, manual
```

#### **LossConfig**
```python
@dataclass
class LossConfig:
    reduction: str = 'mean'
    eps: float = 1e-8
    label_smoothing: float = 0.0
```

### Factory Function
```python
create_loss(
    loss_name: str,
    class_counts: np.ndarray,
    num_classes: int,
    **kwargs
) -> nn.Module
```

### Key Design Principles
- **Unified Interface**: All losses have `forward(logits, target, feature=None)`
- **Memory Efficient**: Lazy weight computation and caching
- **Numerical Stability**: Safe operations with epsilon handling
- **CLI-Friendly**: Both dict-based and keyword argument initialization
- **Composable**: Support for combined losses with custom weights

---

## 5. TRAINING & EVALUATION (train_eval.py)

### Architecture
**File Size:** 4.6 KB (concise!)

### Core Training Function

```python
train_one_epoch(
    model, loader, criterion, optimizer, device,
    logger, epoch, grad_clip=0.0, use_amp=False, scaler=None
)
```

**Features:**
- Mixed-precision training (AMP) with GradScaler
- Gradient clipping
- Non-blocking data transfer (non_blocking=True)
- Per-batch loss computation
- Returns: `{'loss': float, 'acc': float}`

### Evaluation Function

```python
evaluate_with_analysis(
    model, loader, criterion, device, analyzer,
    class_counts: np.ndarray,
    eval_logit_adjust: str,  # 'none'/'posthoc'
    eval_logit_tau: float
)
```

**Returns:**
- Metrics: `{'loss', 'acc', 'balanced_acc'}`
- Analysis: Per-class, per-group metrics (from ClassificationAnalyzer)
- Predictions & Targets
- Timing info (throughput)

### Key Operations
- **Post-hoc Logit Adjustment**: Apply τ*log(π) during evaluation
- **Softmax Probability Extraction**: Used for calibration analysis
- **Feature Extraction**: Handles models that return (logits, features) tuples
- **Flexible Loss Computation**: Tries with features, fallback to logits-only

---

## 6. TWO-STAGE TRAINING (stage2.py)

### Architecture
**File Size:** 4.0 KB (elegant abstraction)

### Stage-2 Modes

#### **CRT (Classifier Re-Training)**
```python
freeze_backbone_params(model, classifier_names)
reinit_classifier_layers(layers)
# Only classifier parameters require_grad=True
```

#### **Fine-Tuning**
```python
unfreeze_all_params(model)
# Differential learning rates for backbone vs. classifier
# Typically: clf_lr_mult=5-10x backbone_lr
```

### Utilities

```python
get_base_model(model)  # Unwrap DataParallel if needed

find_classifier_layers(model, num_classes)
# Auto-detects linear layers with output shape (*, num_classes)

build_stage2_loader(
    dataset, batch_size, num_workers,
    sampler_config,  # {name, alpha, alpha_start, alpha_end}
    total_epochs, seed
)
# Constructs DataLoader with specified sampler strategy

set_batchnorm_eval(module)  # Freeze BN statistics (CRT standard)

apply_tau_norm_to_classifier(layers, tau=1.0)
# Post-hoc weight normalization (Stage-3 calibration)
```

### Key Features
- **Modular Design**: Each operation is independent
- **Sampler Integration**: Seamlessly switches between sampling strategies
- **BN Management**: Proper BN handling for CRT vs. fine-tune modes
- **Flexible Initialization**: Can reinit or keep classifier

---

## 7. OPTIMIZER & SCHEDULER MANAGEMENT (optim_utils.py)

### Architecture
**File Size:** 9.5 KB

### Optimizer Factory
```python
build_optimizer(optimizer_name, params, lr, weight_decay)
# Supports: Adam, SGD, AdamW
```

### Differential Learning Rate Groups
```python
build_optimizer_with_groups(optimizer_name, param_groups, weight_decay)
# For Stage-2 fine-tuning with backbone/classifier different LRs
```

### Scheduler Factory
```python
build_scheduler_for_stage(optimizer, cfg, epochs_this_stage, stage='stage1')
# Supports: cosine, step, plateau, cosine_warmup_restarts
# Auto-handles warmup + after_scheduler composition
```

### Loss Function Builder
```python
build_criterion(loss_name, cfg, class_counts)
# Extracts hyperparameters from config and creates loss instance
# Handles FocalLoss, ClassBalancedLoss, LDAM, LogitAdjustment, Cost-Sensitive, etc.
```

---

## 8. ANALYSIS & METRICS (analysis.py)

### Architecture
**File Size:** 7.6 KB

### ClassificationAnalyzer

```python
analyzer = ClassificationAnalyzer(
    class_counts: np.ndarray,
    grouping: str = 'auto',      # auto/absolute/quantile
    many_thresh: int = 100,
    few_thresh: int = 20,
    q_low: float = 1/3,
    q_high: float = 2/3
)

results = analyzer.analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: Optional[np.ndarray]  # Softmax probabilities
)
```

### Returned Analysis Dictionary

```python
{
    'overall': {
        'accuracy', 'balanced_accuracy', 'macro_precision',
        'macro_recall', 'macro_f1', 'micro_f1', 'gmean_recall',
        'top5', 'macro_auroc', 'macro_auprc'
    },
    'group_wise': {
        'majority': {...},    # Head classes
        'medium': {...},      # Mid classes
        'minority': {...}     # Tail classes
        # Each: accuracy, precision, recall, f1, support, worst_class_recall
    },
    'per_class': {
        'class_0': {precision, recall, f1, support, frequency},
        ...
    },
    'confusion_matrix': [...],
    'confusion_matrix_normalized': [...],
    'worst_class_recall': float,
    'grouping_meta': {...}
}
```

### Key Metrics
- **OA** (Overall Accuracy): Standard accuracy
- **mAcc** (Mean Per-Class Accuracy): Average of per-class recalls
- **Macro-F1**: Mean of per-class F1 scores
- **Balanced Accuracy**: Average of per-class recalls (sklearn standard)
- **Many/Few Acc**: Performance on majority/minority classes
- **HM** (Harmonic Mean): 2*Many*Few/(Many+Few) - balances head/tail

### Grouping Strategies
1. **Absolute**: Fixed thresholds (many_thresh=100, few_thresh=20)
2. **Quantile**: Statistical percentiles (q_low=1/3, q_high=2/3)
3. **Auto**: Selects best based on max class count

---

## 9. CONFIGURATION MANAGEMENT (config.yaml)

### Architecture
**File Size:** 6.7 KB | **Format:** YAML + Hydra/OmegaConf

### Major Sections

```yaml
# Experiment Metadata
exp_name: string
seed: int
device: 'cuda'/'cpu'
gpus: string  # '0,1,2,3'
amp: bool  # Automatic Mixed Precision

# Data Configuration
data:
  path_train: string
  path_val: null or string
  path_test: string
  val_ratio: float  # 0.2 = 20%
  batch_size: int
  num_workers: int
  target_length: int  # 4800 for ADS-B signals
  normalize: bool
  in_memory: bool
  imbalance_ratio: float  # 100.0

# Model Configuration
model:
  name: string  # ConvNetADSB, ResNet1D, MixtureOfExpertsConvNet, etc.
  dropout: float
  use_attention: bool
  norm_kind: string  # auto/bn/gn/ln

# Stage-1: Baseline Training
loss:
  name: string  # CrossEntropy, FocalLoss, ClassBalancedLoss, etc.
  
sampling:
  name: string  # none, inv_freq, sqrt, power, progressive_power
  alpha: float
  alpha_start: float
  alpha_end: float

training:
  epochs: int
  lr: float
  weight_decay: float
  optimizer: string  # Adam, SGD, AdamW
  grad_clip: float
  label_smoothing: float

scheduler:
  name: string  # cosine, step, plateau, cosine_warmup_restarts
  warmup_epochs: int
  warmup_multiplier: float

early_stopping:
  patience: int
  monitor: string  # val_loss, val_acc
  mode: string  # min, max

# Stage-2: Improvement Training (Optional)
stage2:
  enabled: bool
  mode: string  # crt, finetune
  epochs: int
  lr: float
  loss: string
  sampler: string
  freeze_bn: bool
  clf_lr_mult: float

# Stage-3: Post-hoc Calibration
stage3:
  mode: string  # none, tau_norm, logit_adjust, both

# Evaluation
evaluation:
  eval_logit_adjust: string  # none, posthoc
  eval_logit_tau: float

# Visualization
visualization:
  enabled: bool
  dpi: int
  max_samples: int  # For t-SNE
  plot_*: bool  # Individual plot toggles
```

---

## 10. MAIN TRAINING PIPELINE (main.py)

### Architecture
**File Size:** 35 KB | **Lines of Code:** ~900

### Entry Point
```python
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Full training pipeline
```

### Pipeline Stages

1. **Initialization**
   - Seed setup (reproducibility)
   - GPU device setup
   - Directory creation (experiments/logs/checkpoints)

2. **Data Loading**
   - Load train/val/test sets
   - Optional: Create artificial imbalance (long-tail)
   - Build samplers

3. **Model Setup**
   - Instantiate model architecture
   - DataParallel wrapping (multi-GPU)
   - Move to device

4. **Stage-1: Baseline Training** (epochs as specified)
   - Standard training loop with validation
   - Early stopping with best checkpoint
   - Learning rate scheduling (warmup + cosine)

5. **Stage-2: Improvement Training** (Optional)
   - CRT: Freeze backbone, reinit & train classifier only
   - Fine-tune: Differential learning rates (backbone vs classifier)
   - Can use different loss/sampler than Stage-1
   - Best model checkpoint

6. **Stage-3: Calibration** (Optional)
   - τ-normalization of classifier weights
   - Post-hoc logit adjustment

7. **Testing & Analysis**
   - Final test set evaluation
   - Per-class, per-group metrics
   - Confusion matrices

8. **Results & Visualization**
   - Save: config.json, results.json, summary.txt
   - Generate: confusion matrices, ROC curves, calibration plots
   - Create: publication-quality figures

### Key Features
- **Timing Tracking**: Detailed timing for each stage (ms precision)
- **Logging**: Console + file logging with epoch summaries
- **Checkpointing**: Best model auto-saved during training
- **Flexible Config**: All parameters via YAML + CLI overrides
- **Multi-GPU Support**: Automatic DataParallel handling
- **AMP Support**: Mixed-precision training with GradScaler

### CLI Usage
```bash
# Basic
python main.py

# Override config
python main.py exp_name=custom data.batch_size=512 training.lr=5e-4

# Stage-2 with different loss
python main.py stage2.enabled=true stage2.loss=CostSensitiveCE

# Multi-GPU
python main.py gpus="0,1,2,3"
```

---

## 11. VISUALIZATION & ANALYSIS

### visualization.py (49 KB)
Generates publication-quality figures:
- Training curves (loss/accuracy over epochs)
- Learning rate schedules
- Class distribution (grouped by head/medium/tail)
- t-SNE feature embeddings (2D/3D)
- Confusion matrices (raw + normalized)
- Per-class recall bar charts
- Group-wise performance comparison
- Sample count vs. performance scatter plots
- Confidence distribution histograms
- Top-K confusion pairs

### analysis.py (7.6 KB)
Provides detailed classification metrics:
- Per-class: precision, recall, F1, support
- Per-group: head/medium/tail statistics
- Overall: OA, mAcc, Macro-F1, Balanced-Acc, Harmonic Mean
- Advanced: AUROC, AUPRC, G-Mean, Top-5 Accuracy

### summarize.py (19 KB)
Batch experiment comparison:
- Aggregates results from multiple runs
- Creates ranking tables (by OA, mAcc, Macro-F1, Harmonic Mean)
- Comparison matrices (losses vs. samplers)
- Best configuration recommendations
- Improvement over baseline calculation

---

## 12. UTILITY MODULES

### common.py (3.8 KB)
```python
setup_seed(seed)                    # Reproducible randomness
parse_gpu_ids(gpus_input)          # Parse GPU specification
setup_device(which, gpu_ids)       # Device initialization + info

logits_logit_adjustment(logits, class_counts, tau)  # Post-hoc adjustment
tau_norm_weights(weight, tau)      # Classifier weight normalization
```

### training_utils.py (26 KB)
- **GradualWarmupScheduler**: Linear warmup → target LR
- **CosineAnnealingWarmupRestarts**: Cosine annealing with restarts
- **EarlyStopping**: Monitor metric, stop if no improvement
- **ModelCheckpointer**: Save best/periodic checkpoints
- **TrainingManager**: Unified training state management
- **MetricsTracker**: Track loss/accuracy over time

### trainer_logging.py (3.0 KB)
- **TrainingLogger**: Structured logging to file
- Per-epoch summaries: loss, accuracy, LR, group metrics

---

## 13. KEY DESIGN PATTERNS & EXTENSIBILITY POINTS

### Pattern 1: Factory Functions
```python
create_model(name, **kwargs)
create_loss(name, **kwargs)
make_sampler(method, **kwargs)
build_optimizer(name, **kwargs)
build_scheduler_for_stage(...)
```
✅ Easy to add new implementations by extending factory

### Pattern 2: Configuration-Driven Design
```python
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Everything from cfg, no hardcoding
```
✅ Experiment reproducibility, CLI flexibility

### Pattern 3: Modular Loss Functions
```python
class ImbalancedLoss(nn.Module):
    def forward(self, logits, target, feature=None):
        # Standard interface
```
✅ Easy composition (combined losses, conditional application)

### Pattern 4: Sampler Inheritance
```python
class CustomSampler(Sampler):
    def __iter__(self):
        # Custom sampling logic
```
✅ Works seamlessly with PyTorch DataLoader

### Pattern 5: Stage-2 Abstraction
```python
get_base_model()  # Unwrap DataParallel
find_classifier_layers()  # Auto-detect heads
freeze_backbone_params()  # Selective freezing
reinit_classifier_layers()  # Reset head weights
```
✅ CRT/Fine-tune/Custom training modes

---

## 14. ENTRY POINTS FOR EXTENSION

### For Diffusion Models Integration:

1. **Model Addition** (`models.py`)
   - Add diffusion encoder/decoder in `create_model()`
   - Support U-Net, Transformer-based architectures
   - Integrate with existing backbone/classifier interface

2. **Loss Functions** (`imbalanced_losses.py`)
   - Add diffusion-specific losses (score-matching, noise prediction)
   - Keep unified interface `forward(logits/features, target)`
   - Support joint training with classification loss

3. **Training Stages** (`main.py`)
   - Extend Stage-1 for joint training
   - Add Stage-2.5 for diffusion fine-tuning
   - Integrate diffusion-based data augmentation

4. **Data Pipeline** (`data_utils.py`)
   - Add diffusion-based augmentation (`DiffusionAugment`)
   - Support synthetic sample generation for tail classes
   - Implement noisy label handling

5. **Analysis Metrics** (`analysis.py`)
   - Add open-set metrics (AUROC for unknown detection)
   - Confidence calibration analysis
   - Distance-to-decision-boundary visualization

6. **Evaluation** (`train_eval.py`)
   - Add open-set evaluation (unknown rejection)
   - Diffusion likelihood-based confidence
   - Feature anomaly scoring

---

## 15. EXISTING IMPLEMENTATIONS WE CAN BUILD UPON

### Directly Usable Components:

1. **Data Infrastructure**
   - `ADSBSignalDataset`: Easy to add diffusion model inputs
   - Sampling strategies: Can sample tail classes using diffusion
   - Augmentation pipeline: Extensible for diffusion-based augmentation

2. **Model Architecture**
   - Backbone encoders (ConvNetADSB, ResNet1D): Extract features for diffusion
   - Classifier heads: Reuse for open-set classification
   - Normalization factory: Applicable to diffusion modules

3. **Training Framework**
   - Two-stage training: Can adapt for diffusion pre-training + classification fine-tuning
   - Optimizer/scheduler utilities: Reusable for diffusion training
   - Early stopping/checkpointing: Works for any model

4. **Loss Functions**
   - Cost-sensitive losses: Can weight diffusion-generated samples
   - Weighted CE: Natural for importance sampling from diffusion
   - Progressive losses: Can modulate diffusion guidance strength over time

5. **Analysis Pipeline**
   - Per-class metrics: Monitor open-set performance per class
   - Confusion matrix: Track confusion between known classes
   - Group analysis: Compare known vs. unknown class detection

6. **Configuration System**
   - Hydra-based config: Can add diffusion sections seamlessly
   - CLI override system: Easy parameter sweeps
   - Multi-run aggregation: summarize.py handles multiple methods

---

## 16. RUNNING THE EXISTING SYSTEM

### Quick Start
```bash
# Basic training (Stage-1 only)
python main.py exp_name=my_exp

# Full pipeline (Stage-1 + Stage-2 CRT)
python main.py \
  exp_name=full_pipeline \
  stage2.enabled=true \
  stage2.mode=crt \
  stage2.loss=CostSensitiveCE

# Sweep parameter
python main.py exp_name=baseline stage2.enabled=false
python main.py exp_name=improved stage2.enabled=true stage2.mode=crt
python summarize.py  # Compare all runs
```

### Output Structure
```
experiments/
├── my_exp_20250101_120000/
│   ├── logs/
│   │   ├── training.log
│   │   └── console.log
│   ├── results/
│   │   ├── summary.txt
│   │   ├── results.json
│   │   └── plots/
│   ├── checkpoints/
│   │   └── best.pth
│   ├── checkpoints_stage2/
│   │   └── best.pth
│   └── config.json
└── my_exp_latest -> my_exp_20250101_120000/
```

---

## 17. TECHNICAL STACK

**Core Dependencies:**
- PyTorch 1.9+ (torch, torchvision)
- NumPy, Pandas, SciPy
- Hydra, OmegaConf (configuration)
- scikit-learn (metrics)
- h5py (data loading)
- matplotlib, seaborn (visualization)

**Code Quality:**
- Type hints throughout
- Modular design (minimal coupling)
- Comprehensive docstrings
- Error handling with informative messages

---

## 18. SUMMARY TABLE: FILE PURPOSES

| File | Size | Purpose | Key Classes/Functions |
|------|------|---------|----------------------|
| `main.py` | 35 KB | Main training pipeline | `main()` |
| `data_utils.py` | 36 KB | Data loading & sampling | `ADSBSignalDataset`, `make_sampler`, `build_dataloaders` |
| `models.py` | 41 KB | Neural network architectures | `ConvNetADSB`, `ResNet1D`, `DilatedTCN`, `MixtureOfExpertsConvNet` |
| `imbalanced_losses.py` | 50 KB | Loss functions | `create_loss`, `ClassBalancedLoss`, `LDAMLoss`, `CostSensitiveCE` |
| `train_eval.py` | 4.6 KB | Training/eval loops | `train_one_epoch`, `evaluate_with_analysis` |
| `stage2.py` | 4.0 KB | Two-stage training | `freeze_backbone_params`, `build_stage2_loader` |
| `optim_utils.py` | 9.5 KB | Optimizers & schedulers | `build_optimizer`, `build_scheduler_for_stage` |
| `training_utils.py` | 26 KB | Training utilities | `GradualWarmupScheduler`, `EarlyStopping`, `ModelCheckpointer` |
| `analysis.py` | 7.6 KB | Metrics & analysis | `ClassificationAnalyzer` |
| `visualization.py` | 49 KB | Plot generation | `visualize_all_results` |
| `summarize.py` | 19 KB | Experiment comparison | Batch analysis scripts |
| `common.py` | 3.8 KB | Common utilities | `setup_seed`, `parse_gpu_ids`, `logits_logit_adjustment` |
| `trainer_logging.py` | 3.0 KB | Logging | `TrainingLogger` |
| `config.yaml` | 6.7 KB | Default configuration | Hydra configuration |

---

## 19. NEXT STEPS FOR DIFFUSION-BASED OPEN-SET RECOGNITION

Based on this architecture, here are recommended integration points:

1. **Phase 1: Feature Extraction Enhancement**
   - Add diffusion encoder in models.py
   - Use Stage-2 to fine-tune diffusion+classifier jointly

2. **Phase 2: Data Augmentation via Diffusion**
   - Implement conditional diffusion in data_utils.py
   - Generate synthetic samples for tail classes
   - Update sampling strategies to weight diffusion-generated samples

3. **Phase 3: Open-Set Evaluation**
   - Add open-set metrics to analysis.py
   - Implement unknown class rejection in train_eval.py
   - Use diffusion likelihood for confidence scoring

4. **Phase 4: Multi-Stage Diffusion Training**
   - Extend main.py with diffusion pre-training stage
   - Implement diffusion-guided classification fine-tuning
   - Add adversarial unknown generation

5. **Phase 5: Integration Testing**
   - Create config_diffusion.yaml with full pipeline
   - Benchmark vs. baseline
   - Ablation studies using summarize.py

---

## Summary

This codebase is a **well-engineered, production-ready framework** for imbalanced learning that provides an excellent foundation for extending into diffusion-based open-set recognition. The modular design, comprehensive configuration system, and clean abstraction layers make it straightforward to integrate new components while maintaining backward compatibility with existing experiments.

**Key Strengths:**
✅ Modular architecture (easy to extend)
✅ Production quality (proper error handling, logging, checkpointing)
✅ Comprehensive (15+ methods already implemented)
✅ Well-documented (detailed README, clear code)
✅ Reproducible (seed control, configuration versioning)
✅ Scalable (multi-GPU, AMP, lazy loading)

**Ideal Entry Points:**
1. Models → Add diffusion encoder
2. Data → Add diffusion augmentation
3. Losses → Add diffusion-specific objectives
4. Analysis → Add open-set metrics

