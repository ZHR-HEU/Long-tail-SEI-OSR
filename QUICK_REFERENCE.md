# Quick Reference Guide - Long-tail SEI-OSR Codebase

## Visual Directory Tree

```
Long-tail-SEI-OSR/ (7,500 lines, 13 modules)
│
├── [ENTRY POINT]
│   └── main.py (900 lines)
│       ├── Data loading & preprocessing
│       ├── Stage-1: Baseline training (standard CE loss, no re-sampling)
│       ├── Stage-2: Improvement training (CRT or fine-tune)
│       ├── Stage-3: Post-hoc calibration (τ-norm, logit adjustment)
│       └── Evaluation & visualization
│
├── [DATA LOADING & SAMPLING] data_utils.py (36 KB)
│   ├── ADSBSignalDataset
│   ├── make_sampler() factory
│   │   ├── 'none': natural distribution
│   │   ├── 'inv_freq': p∝1/n_c
│   │   ├── 'sqrt': p∝√n_c
│   │   ├── 'power': p∝n_c^α
│   │   └── 'progressive_power': α decays during training
│   ├── Augmentation: TimeShift, Amplitude, Noise, CropOrPad, Normalize
│   └── build_dataloaders()
│
├── [MODELS] models.py (41 KB)
│   ├── Backbones:
│   │   ├── ConvNetADSB (8-layer, 350 channels, default)
│   │   ├── ResNet1D (depth 2-5)
│   │   ├── DilatedTCN (temporal convolution)
│   │   └── Frequency-Domain Experts
│   ├── Classification Heads:
│   │   ├── Linear (standard)
│   │   ├── CosineMarginClassifier (LDAM)
│   │   ├── LogitAdjustedLinear (Menon et al.)
│   │   └── TemperatureScaledClassifier
│   └── Utilities:
│       ├── create_model()
│       ├── EnhancedClassifierInitializer
│       └── swap_classifier()
│
├── [LOSS FUNCTIONS] imbalanced_losses.py (50 KB)
│   ├── Basic: CrossEntropy, FocalLoss
│   ├── Re-weighting: ClassBalancedLoss, LDAMLoss, ProgressiveLoss
│   ├── Post-hoc: BalancedSoftmaxLoss, LogitAdjustmentLoss
│   ├── Cost-Sensitive: CostSensitiveCE, CostSensitiveExpected, CostSensitiveFocal
│   └── create_loss() factory
│
├── [TRAINING INFRASTRUCTURE]
│   ├── train_eval.py (4.6 KB)
│   │   ├── train_one_epoch()
│   │   └── evaluate_with_analysis()
│   ├── stage2.py (4.0 KB)
│   │   ├── get_base_model()
│   │   ├── freeze_backbone_params() / unfreeze_all_params()
│   │   ├── build_stage2_loader()
│   │   └── apply_tau_norm_to_classifier()
│   ├── optim_utils.py (9.5 KB)
│   │   ├── build_optimizer()
│   │   ├── build_scheduler_for_stage()
│   │   └── build_criterion()
│   └── training_utils.py (26 KB)
│       ├── GradualWarmupScheduler
│       ├── CosineAnnealingWarmupRestarts
│       ├── EarlyStopping
│       └── ModelCheckpointer
│
├── [ANALYSIS & VISUALIZATION]
│   ├── analysis.py (7.6 KB)
│   │   └── ClassificationAnalyzer
│   │       ├── Per-class metrics (P/R/F1)
│   │       ├── Per-group metrics (head/medium/tail)
│   │       └── Overall metrics (OA, mAcc, Macro-F1, HM)
│   ├── visualization.py (49 KB)
│   │   └── visualize_all_results()
│   │       ├── Training curves
│   │       ├── t-SNE embeddings
│   │       ├── Confusion matrices
│   │       ├── Per-class recall
│   │       └── Group performance
│   └── summarize.py (19 KB)
│       └── Batch experiment comparison
│
├── [UTILITIES]
│   ├── common.py (3.8 KB)
│   │   ├── setup_seed()
│   │   ├── parse_gpu_ids()
│   │   ├── logits_logit_adjustment()
│   │   └── tau_norm_weights()
│   └── trainer_logging.py (3.0 KB)
│       └── TrainingLogger
│
└── config.yaml (6.7 KB)
    └── Complete experiment configuration template
```

---

## Quick Code Examples

### 1. Run Basic Training
```bash
# Default config
python main.py

# Custom experiment
python main.py exp_name=baseline stage2.enabled=false

# Stage-1 + Stage-2
python main.py \
  exp_name=full \
  stage2.enabled=true \
  stage2.mode=crt \
  stage2.loss=CostSensitiveCE

# Multi-GPU
python main.py gpus="0,1,2,3"
```

### 2. Create Custom Dataset
```python
from data_utils import ADSBSignalDataset, build_dataloaders, LoaderConfig

# Direct instantiation
dataset = ADSBSignalDataset(
    path='data/train.mat',
    target_length=4800,
    normalize=True,
    in_memory=True
)
print(dataset.num_classes, dataset.class_counts)

# Factory method
cfg = LoaderConfig(
    path_train='data/train.mat',
    path_test='data/test.mat',
    batch_size=256,
    sampler='progressive_power'
)
loaders = build_dataloaders(cfg)
train_loader = loaders['train']
```

### 3. Create Custom Model
```python
from models import create_model, swap_classifier
import torch.nn as nn

# Standard models
model = create_model('ConvNetADSB', num_classes=10, dropout_rate=0.1)
model = create_model('ResNet1D', num_classes=10, depth=3)
model = create_model('MixtureOfExpertsConvNet', num_classes=10)

# Change classifier head
from models import ldam_margins_from_counts
margins = ldam_margins_from_counts(class_counts, power=0.25, max_m=0.5)
swap_classifier(model, head='cosine_ldam', class_counts=class_counts, margins=margins)
```

### 4. Use Different Loss Functions
```python
from imbalanced_losses import create_loss
import numpy as np

class_counts = np.array([1000, 500, 100, 50, 10, 5, 2, 1])

# Standard
loss = create_loss('CrossEntropy')

# Focal Loss
loss = create_loss('FocalLoss', gamma=2.0)

# Class-Balanced
loss = create_loss('ClassBalancedLoss', beta=0.9999, class_counts=class_counts)

# LDAM with DRW
loss = create_loss('LDAMLoss', max_margin=0.5, class_counts=class_counts)

# Cost-Sensitive (auto-generates cost matrix)
loss = create_loss('CostSensitiveCE', cost_strategy='auto', class_counts=class_counts)

# In training loop
logits = model(x)
loss_val = loss(logits, target)
```

### 5. Create Custom Sampler
```python
from data_utils import make_sampler

labels = np.array([0]*1000 + [1]*100 + [2]*10)  # Imbalanced

# Different strategies
sampler_inv = make_sampler(labels, method='inv_freq')              # p∝1/n_c
sampler_sqrt = make_sampler(labels, method='sqrt')                 # p∝√n_c
sampler_power = make_sampler(labels, method='power', alpha=0.5)    # p∝n_c^0.5
sampler_prog = make_sampler(
    labels, 
    method='progressive_power',
    alpha_start=0.5,
    alpha_end=0.0,
    total_epochs=100
)

# Use in DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, sampler=sampler_prog, batch_size=256)
```

### 6. Two-Stage Training
```python
from stage2 import (
    get_base_model, find_classifier_layers, 
    freeze_backbone_params, reinit_classifier_layers,
    set_batchnorm_eval, build_stage2_loader
)

# Stage-1 training (standard)
# ... train model on train_loader

# Stage-2: CRT mode
base = get_base_model(model)
clf_pairs = find_classifier_layers(base, num_classes=10)
classifier_names = [n for n, _ in clf_pairs]

# Freeze backbone
freeze_backbone_params(base, classifier_names)
# Reinit classifier
reinit_classifier_layers([m for _, m in clf_pairs])
# Freeze BN (standard for CRT)
base.apply(set_batchnorm_eval)

# Create Stage-2 loader with different sampler
stage2_loader = build_stage2_loader(
    train_dataset,
    batch_size=256,
    num_workers=4,
    sampler_config={'name': 'progressive_power', 'alpha_start': 0.5, 'alpha_end': 0.0},
    total_epochs=100
)

# Stage-2 training (only classifier parameters updated)
for epoch in range(100):
    # train_one_epoch with stage2_loader
    pass
```

### 7. Analyze Results
```python
from analysis import ClassificationAnalyzer

# Create analyzer
class_counts = np.array([1000, 500, 100, 50, 10])
analyzer = ClassificationAnalyzer(
    class_counts=class_counts,
    grouping='auto'  # or 'absolute', 'quantile'
)

# Analyze predictions
y_true = np.array([0, 1, 2, 1, 0, 3, 2, 4])
y_pred = np.array([0, 1, 2, 1, 0, 3, 1, 4])
probs = ... # softmax output (optional)

results = analyzer.analyze_predictions(y_true, y_pred, prob=probs)

# Access results
print("Overall Accuracy:", results['overall']['accuracy'])
print("Mean Per-Class Acc:", results['overall']['macro_recall'])
print("Macro-F1:", results['overall']['macro_f1'])
print("\nHead Classes:", results['group_wise']['majority'])
print("Tail Classes:", results['group_wise']['minority'])
print("Per-class metrics:", results['per_class'])
```

### 8. Evaluate with Analysis
```python
from train_eval import evaluate_with_analysis

val_metrics, analysis, predictions, targets, timing = evaluate_with_analysis(
    model=model,
    loader=val_loader,
    criterion=criterion,
    device=device,
    analyzer=analyzer,
    class_counts=class_counts,
    eval_logit_adjust='posthoc',  # 'none' or 'posthoc'
    eval_logit_tau=1.0
)

print(f"Loss: {val_metrics['loss']:.4f}")
print(f"Accuracy: {val_metrics['acc']:.2f}%")
print(f"Balanced Acc: {val_metrics['balanced_acc']:.2f}%")
print(f"Throughput: {timing['throughput_samples_per_sec']:.2f} samples/sec")
```

### 9. Custom Configuration (CLI)
```bash
# Override any parameter
python main.py \
  exp_name=custom_exp \
  data.batch_size=512 \
  data.imbalance_ratio=50 \
  model.name=ResNet1D \
  model.dropout=0.2 \
  training.epochs=300 \
  training.lr=5e-4 \
  loss.name=LDAMLoss \
  sampling.name=progressive_power \
  stage2.enabled=true \
  stage2.mode=crt \
  stage2.loss=CostSensitiveCE \
  visualization.enabled=true
```

### 10. Batch Experiment Comparison
```bash
# Run multiple experiments
python main.py exp_name=exp1 loss.name=CrossEntropy stage2.enabled=false
python main.py exp_name=exp2 loss.name=FocalLoss stage2.enabled=false
python main.py exp_name=exp3 loss.name=ClassBalancedLoss stage2.enabled=true
python main.py exp_name=exp4 loss.name=LDAMLoss stage2.enabled=true

# Compare all results
python summarize.py

# Outputs:
# - analysis_results_*/REPORT.md
# - analysis_results_*/ranking_overall.csv
# - analysis_results_*/comparison_loss.csv
# - analysis_results_*/matrix_*.csv
```

---

## Key Class Initialization Patterns

### Pattern 1: Model Creation
```python
model = create_model(
    'ConvNetADSB',              # Model name
    num_classes=10,             # Number of output classes
    dropout_rate=0.1,           # Dropout probability
    use_attention=True,         # Channel attention
    norm_kind='auto'            # BatchNorm/GroupNorm/LayerNorm
)
```

### Pattern 2: Loss Creation
```python
loss = create_loss(
    'CostSensitiveCE',          # Loss function name
    class_counts=class_counts,  # Class frequencies (for weighting)
    cost_strategy='auto',       # How to compute cost matrix
    reduction='mean'            # Loss reduction method
)
```

### Pattern 3: Sampler Creation
```python
sampler = make_sampler(
    labels=train_labels,        # Labels of training set
    method='progressive_power',  # Sampling strategy
    seed=42,                    # Reproducibility
    alpha_start=0.5,            # Initial alpha
    alpha_end=0.0,              # Final alpha
    total_epochs=200            # For decay schedule
)
```

### Pattern 4: Optimizer & Scheduler
```python
# Optimizer
optimizer = build_optimizer(
    'Adam',                     # Optimizer name
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# Scheduler
scheduler = build_scheduler_for_stage(
    optimizer, cfg, 
    epochs_this_stage=200,
    stage='stage1'
)
```

---

## Configuration Hierarchy

```
config.yaml (base configuration)
    ↓
Hydra CLI overrides (python main.py key=value)
    ↓
main.py (instantiates all components from config)
    ↓
Component initialization (models, losses, optimizers, etc.)
```

### Config Structure
```yaml
# High level
exp_name, seed, gpus, amp

# Data: ADSBSignalDataset parameters
data:
  path_train, path_test, batch_size, num_workers, target_length, normalize, imbalance_ratio

# Model: create_model() parameters
model:
  name, dropout, use_attention, norm_kind

# Stage-1: Training loop
loss:
  name, [other_params]
sampling:
  name, alpha, alpha_start, alpha_end
training:
  epochs, lr, weight_decay, optimizer, grad_clip
scheduler:
  name, warmup_epochs, warmup_multiplier

# Stage-2: Optional improvement training
stage2:
  enabled, mode (crt/finetune), loss, sampler, freeze_bn, clf_lr_mult

# Stage-3: Optional calibration
stage3:
  mode (none/tau_norm/logit_adjust/both)

# Evaluation & Visualization
evaluation:
  eval_logit_adjust, eval_logit_tau
visualization:
  enabled, [plot_options]
```

---

## Metric Definitions

| Metric | Formula | Use Case |
|--------|---------|----------|
| **OA** | Accuracy | Overall performance (can be misleading on imbalanced data) |
| **mAcc** | mean(recall_c) | Mean per-class recall (key metric for imbalanced learning) |
| **Macro-F1** | mean(F1_c) | Balanced precision+recall average |
| **Balanced-Acc** | mean(recall_c) | Same as mAcc (sklearn name) |
| **G-Mean** | exp(mean(log(recall_c))) | Geometric mean of recalls |
| **Many-Acc** | Recall on majority classes | Head class performance |
| **Few-Acc** | Recall on minority classes | Tail class performance |
| **HM** | 2*Many*Few/(Many+Few) | Harmonic mean (balances head/tail) |
| **Macro-AUC** | OvR ROC-AUC average | Discriminability per class |

**For Papers:**
- Always report: OA, mAcc, Many-Acc, Few-Acc, HM
- Also report: Macro-F1, Balanced-Acc
- For specific scenarios: G-Mean, Macro-AUC

---

## File Dependencies Graph

```
main.py (orchestrator)
    ├── data_utils.py (data loading & sampling)
    ├── models.py (model creation)
    ├── imbalanced_losses.py (loss creation)
    ├── train_eval.py (training loops)
    │   └── analysis.py (metric computation)
    ├── stage2.py (two-stage training)
    ├── optim_utils.py (optimizer/scheduler building)
    ├── training_utils.py (utilities)
    ├── analysis.py (detailed analysis)
    ├── visualization.py (plot generation)
    ├── trainer_logging.py (logging)
    ├── common.py (utilities)
    └── config.yaml (configuration)
```

**Minimal Dependencies** (independent modules):
- `common.py` - Depends only on PyTorch, NumPy
- `analysis.py` - Depends on sklearn
- `models.py` - Depends on PyTorch
- `imbalanced_losses.py` - Depends on PyTorch, NumPy

---

## Extension Checklist for Diffusion Integration

- [ ] **Add diffusion model in models.py**
  - [ ] DiffusionUNet (time-domain)
  - [ ] DiffusionTransformer (alternative)
  - [ ] Integrate with existing backbone interfaces

- [ ] **Add diffusion loss in imbalanced_losses.py**
  - [ ] Score matching loss
  - [ ] Denoising loss
  - [ ] Joint classification+diffusion loss

- [ ] **Add diffusion augmentation in data_utils.py**
  - [ ] DiffusionAugment class
  - [ ] Conditional generation for tail classes
  - [ ] Sampling weight adjustment

- [ ] **Extend main.py**
  - [ ] Diffusion pre-training stage
  - [ ] Diffusion-guided classification fine-tuning
  - [ ] Unknown generation for open-set

- [ ] **Add open-set metrics to analysis.py**
  - [ ] AUROC for unknown detection
  - [ ] Average precision for tail classes
  - [ ] Confidence calibration

- [ ] **Extend evaluation in train_eval.py**
  - [ ] Unknown class rejection
  - [ ] Diffusion likelihood scoring
  - [ ] Feature anomaly detection

- [ ] **Create config_diffusion.yaml**
  - [ ] Diffusion model parameters
  - [ ] Joint training settings
  - [ ] Open-set evaluation settings

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size, num_workers, max_samples for visualization |
| Slow data loading | Set in_memory=True, increase num_workers |
| Poor tail class performance | Use progressive_power sampler, increase stage2 epochs |
| Overfitting on small dataset | Increase dropout, weight_decay, use more augmentation |
| NaN loss during training | Check class_counts normalization, reduce learning rate |
| Conflicting logit adjustment | Only apply once: in loss OR in evaluation, not both |

---

## Performance Benchmarks (Reference)

Typical timings on ADS-B signal classification (8 classes, 1000-20000 samples):

| Operation | Time (per epoch) |
|-----------|-----------------|
| Stage-1 training (200 epochs) | 2-5 minutes |
| Stage-2 CRT (100 epochs) | 1-3 minutes |
| Validation (full dataset) | 10-30 seconds |
| Test evaluation | 5-10 seconds |
| Visualization (with t-SNE) | 2-5 minutes |
| Total pipeline | 5-15 minutes |

With multi-GPU (4x GPUs): Expect 3-4x speedup

---

## References & Links

**Papers Implemented:**
1. Cui et al., CVPR 2019 - Class-Balanced Loss
2. Cao et al., NeurIPS 2019 - LDAM Loss
3. Menon et al., ICLR 2021 - Logit Adjustment
4. Kang et al., ICLR 2020 - Decoupling Representation & Classifier

**Configuration Framework:**
- Hydra: https://hydra.cc/
- OmegaConf: https://omegaconf.readthedocs.io/

**PyTorch Documentation:**
- DataLoader & Sampler: https://pytorch.org/docs/stable/data.html
- Distributed Training: https://pytorch.org/docs/stable/nn.html#dataparallel

---

## Final Notes

1. **Configuration is King**: Almost everything is controlled via config.yaml - easy to experiment
2. **Multi-Stage Design**: Stage-1 (baseline) → Stage-2 (improvement) → Stage-3 (calibration)
3. **Flexibility**: Mix-and-match losses, samplers, and models freely
4. **Reproducibility**: Always set seed for deterministic results
5. **Extensibility**: Factory functions make it easy to add new components
6. **Scalability**: Supports single-GPU to multi-GPU training seamlessly

For questions, refer to:
- README.md (comprehensive documentation)
- Code docstrings (inline documentation)
- config.yaml comments (parameter explanations)

