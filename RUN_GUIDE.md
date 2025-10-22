# 🚀 长尾开集识别两阶段训练运行指南

## 📋 目录
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [运行命令](#运行命令)
- [参数调整指南](#参数调整指南)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 🎯 快速开始

### 1. 环境要求

```bash
# Python 3.8+
# PyTorch 1.10+
# CUDA 11.3+ (可选，用于GPU加速)

# 确保已安装所需依赖
pip install -r requirements.txt  # 如果有的话
```

### 2. 检查数据路径

确保配置文件中的数据路径正确：

```bash
# 查看当前配置的数据路径
grep "path_train\|path_test" config_openset_twostage.yaml
```

当前配置:
```yaml
path_train: "/home/dell/md3/zhahaoran/data/ADS-B_Train_100X360-2_5-10-15-20dB.mat"
path_test: "/home/dell/md3/zhahaoran/data/ADS-B_test_100X40_5-10-15-20dB.mat"
```

### 3. 运行两阶段训练

```bash
# 基础运行（使用默认配置）
python train_openset_twostage.py

# 指定配置文件运行
python train_openset_twostage.py --config config_openset_twostage.yaml
```

---

## ⚙️ 配置文件说明

### 主要配置文件

| 文件名 | 用途 | 说明 |
|-------|------|------|
| `config_openset_twostage.yaml` | **两阶段训练** | 推荐用于长尾开集识别 |
| `config_openset.yaml` | 单阶段训练 | 简化版本，快速实验 |
| `config.yaml` | 基础闭集训练 | 不包含开集检测 |

### config_openset_twostage.yaml 核心参数

```yaml
# ============================================================
# 两阶段训练核心配置
# ============================================================

# 实验配置
exp_name: "openset_twostage_crt"  # 实验名称（会自动创建目录）
seed: 42                            # 随机种子
device: "cuda"                      # cuda 或 cpu

# 数据配置
data:
  path_train: "/path/to/train.mat"  # 训练数据路径
  path_test: "/path/to/test.mat"    # 测试数据路径
  num_known_classes: 6              # 已知类别数量
  split_protocol: "random"          # 类别划分方式
  imbalance_ratio: 100.0            # 长尾不平衡比例
  batch_size: 256
  num_workers: 4

# Stage 1: 表示学习（闭集）
stage1:
  enabled: true
  epochs: 200
  sampling_strategy: "none"         # 自然分布
  loss:
    name: "Focal"                   # CrossEntropy | Focal | LDAM
  diffusion:
    enabled: false                  # 是否使用扩散模型

# Stage 2: 分类器重训练 + 开集检测
stage2:
  enabled: true
  epochs: 200
  mode: "crt"                       # crt（冻结backbone） | finetune
  sampling_strategy: "progressive_power"  # 渐进式重采样
  loss:
    name: "CostSensitiveCE"         # 代价敏感损失
  openset:
    detector_type: "openmax"        # openmax | odin | energy | mahalanobis
    refit_interval: 20              # 每N个epoch重新拟合检测器
```

---

## 🎮 运行命令详解

### 方式1: 使用默认配置（推荐新手）

```bash
python train_openset_twostage.py
```

**说明**: 自动使用 `config_openset_twostage.yaml`

### 方式2: 指定配置文件

```bash
python train_openset_twostage.py --config config_openset_twostage.yaml
```

### 方式3: 仅运行Stage-1（测试backbone）

编辑 `config_openset_twostage.yaml`:
```yaml
stage1:
  enabled: true
  epochs: 100

stage2:
  enabled: false  # 禁用Stage-2
```

然后运行:
```bash
python train_openset_twostage.py --config config_openset_twostage.yaml
```

### 方式4: 后台运行（长时间训练）

```bash
# 使用nohup后台运行
nohup python train_openset_twostage.py > training.log 2>&1 &

# 查看实时日志
tail -f training.log

# 查看进程
ps aux | grep train_openset_twostage
```

### 方式5: 使用tmux/screen（推荐）

```bash
# 创建新会话
tmux new -s training

# 运行训练
python train_openset_twostage.py

# 分离会话: Ctrl+B 然后按 D

# 重新连接
tmux attach -t training
```

---

## 🎛️ 参数调整指南

### 场景1: 数据集较小（< 1万样本）

```yaml
data:
  batch_size: 128              # 减小batch size

stage1:
  epochs: 100                  # 减少epochs
  early_stopping_patience: 20

stage2:
  epochs: 100
```

### 场景2: 数据集很大（> 10万样本）

```yaml
data:
  batch_size: 512              # 增大batch size
  num_workers: 8               # 增加数据加载线程

stage1:
  epochs: 300
  lr: 5e-4                     # 可能需要调小学习率
```

### 场景3: GPU显存不足

```yaml
data:
  batch_size: 64               # 减小batch size

model:
  dropout: 0.5                 # 增大dropout以防过拟合

training:
  amp: true                    # 启用混合精度训练（如果支持）
```

### 场景4: 追求最高性能

```yaml
stage1:
  epochs: 300
  sampling_strategy: "sqrt"    # 使用平方根采样
  loss:
    name: "LDAM"               # 使用LDAM损失

stage2:
  mode: "crt"                  # CRT模式通常最好
  sampling_strategy: "progressive_power"
  loss:
    name: "CostSensitiveCE"
  openset:
    detector_type: "openmax"   # OpenMax通常效果最好
    openmax:
      tailsize: 50             # 增加尾部样本数
```

### 场景5: 快速实验/调试

```yaml
stage1:
  epochs: 10                   # 快速跑几轮看看
  early_stopping_patience: 5

stage2:
  epochs: 10

visualization:
  enabled: false               # 关闭可视化加速
```

---

## 📊 输出说明

### 目录结构

运行后会自动创建以下目录结构:

```
./checkpoints/
├── stage1/
│   ├── best_model.pth        # Stage-1最佳模型
│   ├── checkpoint_epoch_50.pth
│   └── training_history.json
│
└── stage2/
    ├── best_model.pth        # Stage-2最佳模型（最终模型）
    ├── checkpoint_epoch_50.pth
    ├── final_results.txt     # 最终测试结果
    └── training_history.json
```

### 日志输出示例

```
================================================================================
                Two-Stage Open-Set Training Configuration
================================================================================
exp_name: openset_twostage_crt
seed: 42
device: cuda
...

================================================================================
                            STAGE 1: Representation Learning (Closed-Set)
================================================================================

Creating Model...
Model created: ConvNetADSB
Feature dimension: 256

Training Stage-1 (Closed-Set Classification Only)
--------------------------------------------------------------------------------
Epoch [1/200] | Loss: 2.1543 | Acc: 45.32% | Val Loss: 1.8765 | Val Acc: 52.18%
...

[Stage-1 Completed Successfully]
Best model saved in: ./checkpoints/stage1

================================================================================
                      STAGE 2: Classifier Retraining + Open-Set Detection
================================================================================

Preparing model for Stage-2...
Mode: crt
[CRT Mode]
  - Freezing backbone parameters
  - Reinitializing classifier layers

Fitting Open-Set Detector on Stage-1 Features
--------------------------------------------------------------------------------
Detector type: openmax
Fitting OpenMax detector...

Training Stage-2 (Classifier Retraining + Open-Set Detection)
--------------------------------------------------------------------------------
Epoch [1/200] | Loss: 1.5432 | Acc: 68.45% | AUROC: 0.82 | OSCR: 0.75
...

================================================================================
                           Final Evaluation on Test Set
================================================================================

Closed-Set Performance:
  Closed-Set Accuracy: 0.7856
  Overall Accuracy: 0.7234

Open-Set Performance:
  AUROC: 0.8912
  AUPR: 0.8654
  FPR95: 0.1234
  OSCR: 0.8123
  F1-Score: 0.7845

Long-Tail Analysis:
  Many-shot Acc: 0.8945
  Medium-shot Acc: 0.7523
  Few-shot Acc: 0.6234

Results saved to: ./checkpoints/stage2/final_results.txt

[Stage-2 Completed Successfully]

================================================================================
           Two-Stage Training Pipeline Completed Successfully!
================================================================================
```

### 关键指标说明

| 指标 | 说明 | 目标值 |
|-----|------|-------|
| **Closed-Set Acc** | 已知类准确率 | > 75% |
| **AUROC** | 开集检测ROC曲线下面积 | > 85% |
| **OSCR** | 开集分类准确率 | > 75% |
| **FPR95** | 95%召回率时的误报率 | < 15% |
| **Many/Few-shot Acc** | 头部/尾部类准确率 | 差距 < 20% |

---

## 🔧 常见问题

### Q1: 提示找不到数据文件

**错误**: `FileNotFoundError: /path/to/train.mat`

**解决方法**:
```bash
# 1. 检查数据文件是否存在
ls -lh /home/dell/md3/zhahaoran/data/*.mat

# 2. 如果路径不对，修改配置文件
vim config_openset_twostage.yaml
# 更新 data.path_train 和 data.path_test

# 3. 或者使用软链接
ln -s /实际路径/data /home/dell/md3/zhahaoran/data
```

### Q2: CUDA out of memory

**错误**: `RuntimeError: CUDA out of memory`

**解决方法**:
```yaml
# 在配置文件中调整:
data:
  batch_size: 64        # 减小到64或更小
  num_workers: 2        # 减少工作线程
```

或使用CPU:
```yaml
device: "cpu"
```

### Q3: ValueError: Unknown norm kind

**错误**: `ValueError: Unknown norm kind: batch_norm`

**解决方法**:
```yaml
# 在配置文件中修改 norm_kind
model:
  norm_kind: "auto"  # 支持的值: "auto", "bn", "gn", "ln"
```

**说明**: 模型只支持以下norm类型：
- `"auto"`: 自动选择（默认为BatchNorm）
- `"bn"`: BatchNorm1d
- `"gn"`: GroupNorm
- `"ln"`: LayerNorm

### Q4: ValueError: diffusion_model must be provided when use_diffusion=True

**错误**: `ValueError: diffusion_model and features must be provided when use_diffusion=True`

**原因**: Loss配置中启用了diffusion loss，但diffusion模型被禁用或未创建。

**解决方法**:

**选项1 - 禁用diffusion loss（推荐用于Stage-1）**:
```yaml
stage1:
  loss:
    loss_type: "focal"
    use_diffusion: false      # ← 确保这是 false
    use_contrastive: false
    use_entropy: false
```

**选项2 - 启用diffusion模型（如需使用diffusion loss）**:
```yaml
stage1:
  loss:
    use_diffusion: true
    lambda_diffusion: 0.1

  diffusion:
    enabled: true             # ← 启用diffusion模型
    hidden_dims: [512, 256]
    timesteps: 1000
```

**说明**:
- Stage-1通常只需要基础分类损失，不需要diffusion
- 如果要使用diffusion loss，必须同时启用diffusion模型
- 确保配置文件格式正确（见下方完整示例）

**完整的loss配置格式**:
```yaml
loss:
  loss_type: "focal"          # 基础损失: ce, focal, ldam, balanced_softmax, cb
  use_diffusion: false        # 是否使用diffusion loss
  use_contrastive: false      # 是否使用对比学习loss
  use_entropy: false          # 是否使用熵正则
  use_objectosphere: false    # 是否使用objectosphere loss

  # Loss权重（如果启用相应loss）
  lambda_diffusion: 0.1
  lambda_contrastive: 0.1
  lambda_entropy: 0.01
  lambda_objectosphere: 0.1

  # 基础loss的参数
  gamma: 2.0                  # Focal loss参数
  alpha: 0.25
```

### Q5: ImportError: No module named 'xxx'

**解决方法**:
```bash
# 安装缺失的包
pip install scipy h5py pyyaml scikit-learn

# 如果是conda环境
conda install scipy h5py pyyaml scikit-learn
```

### Q6: Stage-1训练很慢

**可能原因和解决方法**:

1. **数据加载慢**:
```yaml
data:
  num_workers: 8       # 增加工作线程
  pin_memory: true     # 启用内存固定
```

2. **模型太大**:
```yaml
model:
  dropout: 0.3         # 减小模型复杂度
```

3. **使用CPU训练**:
```yaml
device: "cuda"         # 改用GPU
```

### Q7: 开集检测效果不好 (AUROC < 0.7)

**调优建议**:

```yaml
# 1. 尝试不同的检测器
stage2:
  openset:
    detector_type: "openmax"  # 尝试: openmax, odin, energy

# 2. 增加Stage-1训练时间
stage1:
  epochs: 300
  early_stopping_patience: 50

# 3. 启用扩散模型增强
stage1:
  diffusion:
    enabled: true

stage2:
  diffusion:
    enabled: true

# 4. 调整重采样策略
stage2:
  sampling_strategy: "class_uniform"  # 尝试完全平衡采样
```

### Q8: 尾部类别准确率很低

**调优建议**:

```yaml
# 1. 使用更激进的重采样
stage2:
  sampling_strategy: "class_uniform"  # 类均匀采样

# 2. 使用代价敏感损失
stage2:
  loss:
    name: "CostSensitiveCE"

# 3. 增加Stage-2训练时间
stage2:
  epochs: 300
```

### Q9: 如何可视化训练过程？

```bash
# 1. 实时监控训练日志
tail -f training.log

# 2. 使用TensorBoard（如果启用）
tensorboard --logdir ./checkpoints

# 3. 训练结束后查看结果文件
cat ./checkpoints/stage2/final_results.txt
```

### Q10: 如何恢复中断的训练？

目前代码不支持自动恢复，但可以手动调整：

```yaml
# 如果Stage-1已完成，直接从Stage-2开始
stage1:
  enabled: false          # 禁用Stage-1
  checkpoint_dir: "./checkpoints/stage1"  # 使用已有模型

stage2:
  enabled: true
  load_stage1_checkpoint: "./checkpoints/stage1/best_model.pth"
```

---

## 🎯 推荐配置模板

### 模板1: 标准配置（平衡性能和速度）

```yaml
exp_name: "openset_standard"
seed: 42
device: "cuda"

data:
  num_known_classes: 6
  imbalance_ratio: 100.0
  batch_size: 256

stage1:
  epochs: 200
  sampling_strategy: "none"
  loss:
    name: "Focal"

stage2:
  epochs: 200
  mode: "crt"
  sampling_strategy: "progressive_power"
  loss:
    name: "CostSensitiveCE"
  openset:
    detector_type: "openmax"
```

### 模板2: 高性能配置（追求最佳效果）

```yaml
exp_name: "openset_high_performance"
seed: 42
device: "cuda"

data:
  num_known_classes: 6
  imbalance_ratio: 100.0
  batch_size: 512

stage1:
  epochs: 300
  sampling_strategy: "sqrt"
  loss:
    name: "LDAM"
  diffusion:
    enabled: true

stage2:
  epochs: 300
  mode: "crt"
  sampling_strategy: "progressive_power"
  loss:
    name: "CostSensitiveCE"
  openset:
    detector_type: "openmax"
    openmax:
      tailsize: 50
    refit_interval: 10
```

### 模板3: 快速实验配置

```yaml
exp_name: "openset_quick"
seed: 42
device: "cuda"

data:
  batch_size: 128

stage1:
  epochs: 50
  early_stopping_patience: 10

stage2:
  epochs: 50
  early_stopping_patience: 10
```

---

## 📚 更多资源

- **代码文档**: 查看各个模块的docstring
- **原理说明**: 参考论文或README.md
- **问题反馈**: 提交GitHub Issue

---

## 🚀 开始训练！

```bash
# 1. 确认配置
cat config_openset_twostage.yaml | grep -A 3 "data:"

# 2. 开始训练
python train_openset_twostage.py

# 3. 监控进度（另开终端）
watch -n 5 "tail -20 checkpoints/stage2/training_history.json"

# 4. 查看结果
cat checkpoints/stage2/final_results.txt
```

**祝训练顺利！** 🎉
