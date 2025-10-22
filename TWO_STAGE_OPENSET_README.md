# Two-Stage Open-Set Training for Long-Tail Recognition

## 概述

本实现提供了一个**两阶段训练框架**，用于长尾开集识别任务。将特征学习和开集检测分离，提高了模型的稳定性和性能。

## 架构设计

### Stage 1: 表示学习（闭集）
- **目标**：在已知类上学习稳健的特征表示
- **训练内容**：
  - 闭集分类任务
  - 可选的扩散模型用于特征增强
  - 专注于特征质量，不涉及开集检测
- **损失函数**：长尾损失（Focal/LDAM/BalancedSoftmax等）
- **采样策略**：自然分布或轻度平衡
- **输出**：训练好的backbone + 分类器

### Stage 2: 分类器重训练 + 开集检测
- **目标**：平衡长尾分布 + 开集未知类检测
- **两种模式**：
  1. **CRT模式（推荐）**：
     - 冻结backbone
     - 重新初始化分类器
     - 在稳定特征空间上训练
  2. **Fine-tuning模式**：
     - Backbone使用小学习率
     - Classifier使用大学习率
     - 整体微调

- **训练内容**：
  - 重采样/重加权（Progressive Power Sampling）
  - 拟合开集检测器（OpenMax/ODIN/Energy等）
  - 定期重新拟合检测器

- **关键优势**：
  - 特征稳定后再拟合检测器，检测更准确
  - 分类器重训练专注于长尾平衡
  - 检测器在稳定特征空间上工作，泛化性更好

## 使用方法

### 1. 配置文件

配置文件位于：`config_openset_twostage.yaml`

**关键配置项**：

```yaml
# Stage 1配置
stage1:
  enabled: true
  epochs: 200
  sampling_strategy: "none"        # 自然分布
  loss:
    name: "Focal"                  # 长尾损失
  diffusion:
    enabled: false                 # 是否使用扩散模型

# Stage 2配置
stage2:
  enabled: true
  epochs: 200
  mode: "crt"                      # "crt" 或 "finetune"
  bn_mode: "freeze"                # BN统计量冻结
  sampling_strategy: "progressive_power"  # 渐进式采样
  alpha_start: 0.5
  alpha_end: 0.0
  loss:
    name: "CostSensitiveCE"        # 代价敏感交叉熵
  openset:
    detector_type: "openmax"       # 开集检测器类型
    refit_interval: 20             # 重新拟合间隔
```

### 2. 运行训练

```bash
# 使用默认配置
python train_openset_twostage.py

# 使用自定义配置
python train_openset_twostage.py --config my_config.yaml
```

### 3. 输出结果

训练完成后，会生成以下输出：

```
checkpoints/
├── stage1/
│   ├── best_model.pth          # Stage-1最佳模型
│   ├── training.log            # Stage-1训练日志
│   └── checkpoint_epoch_*.pth  # 定期检查点
└── stage2/
    ├── best_model.pth          # Stage-2最佳模型
    ├── training.log            # Stage-2训练日志
    ├── final_results.txt       # 最终测试结果
    └── checkpoint_epoch_*.pth  # 定期检查点
```

## 配置建议

### 对于不同的场景

#### 1. 极端长尾（IR=100+）
```yaml
stage1:
  loss:
    name: "LDAM"                # 大间隔损失
  sampling_strategy: "none"     # 自然分布，让模型先学习表示

stage2:
  mode: "crt"                   # 冻结backbone
  sampling_strategy: "progressive_power"
  alpha_start: 0.5
  alpha_end: 0.0
  loss:
    name: "CostSensitiveCE"
```

#### 2. 中等长尾（IR=10-50）
```yaml
stage1:
  loss:
    name: "Focal"               # Focal loss
  sampling_strategy: "sqrt"     # 平方根采样

stage2:
  mode: "finetune"              # 微调模式
  lr_backbone: 1e-5
  lr_classifier: 1e-3
  loss:
    name: "CrossEntropy"
```

#### 3. 强调开集检测性能
```yaml
stage1:
  diffusion:
    enabled: true               # 启用扩散模型
    loss_weight: 0.1

stage2:
  openset:
    detector_type: "openmax"    # 或 "odin", "energy"
    refit_interval: 10          # 更频繁地重新拟合
```

## 开集检测器选择

### OpenMax（推荐）
- **原理**：极值理论 + Weibull分布
- **优点**：理论基础强，适合长尾
- **配置**：
  ```yaml
  openmax:
    tailsize: 20
    alpharank: 3
    distance_metric: "cosine"
  ```

### ODIN
- **原理**：温度缩放 + 输入扰动
- **优点**：对深度网络效果好
- **配置**：
  ```yaml
  odin:
    temperature: 1000.0
    epsilon: 0.0012
  ```

### Energy-based
- **原理**：自由能作为不确定性度量
- **优点**：简单高效
- **配置**：
  ```yaml
  energy:
    temperature: 1.0
  ```

## 性能指标

训练过程会监控以下指标：

**闭集性能**：
- Closed-Set Accuracy: 已知类分类准确率
- Overall Accuracy: 整体准确率（包括未知类）
- Many/Medium/Few-shot Acc: 头部/中部/尾部类准确率

**开集性能**：
- AUROC: ROC曲线下面积
- AUPR: PR曲线下面积
- OSCR: 开集分类率
- FPR95: 95%召回率下的误报率

## 与单阶段训练的对比

| 特性 | 单阶段训练 | 两阶段训练 |
|------|----------|----------|
| **特征稳定性** | 特征一直在变化 | Stage-2特征稳定 |
| **检测器拟合** | 在变化特征上拟合 | 在稳定特征上拟合 |
| **长尾处理** | 训练时处理 | Stage-2专门处理 |
| **开集性能** | 中等 | 更好 |
| **训练时间** | 较短 | 较长（两倍） |
| **推荐场景** | 轻度长尾 | 极端长尾+开集 |

## 常见问题

### Q1: 什么时候使用CRT模式？什么时候使用Fine-tuning模式？

**CRT模式**（推荐）：
- 极端长尾（IR > 50）
- Stage-1已经学到了很好的特征
- 想要快速收敛

**Fine-tuning模式**：
- 中等长尾（IR < 50）
- 想要进一步优化特征
- 有足够的训练时间

### Q2: 是否应该在Stage-1启用扩散模型？

**建议**：
- 如果数据量大且计算资源充足：启用
- 如果想要最佳开集检测性能：启用
- 如果训练时间有限：禁用

扩散模型主要用于特征空间的增强，可以提升开集检测性能，但会增加训练时间。

### Q3: 检测器重新拟合间隔如何设置？

**建议**：
- CRT模式：20-30 epochs（特征变化较小）
- Fine-tuning模式：10-15 epochs（特征仍在变化）

### Q4: 如何选择开集检测器？

**推荐顺序**：
1. **OpenMax**：通用推荐，适合长尾
2. **Energy-based**：简单高效，适合快速实验
3. **ODIN**：需要调参，但效果可能更好
4. **Mahalanobis**：需要较多样本，适合数据量大的场景

## 代码结构

```
train_openset_twostage.py       # 主训练程序
config_openset_twostage.yaml    # 配置文件

# 依赖的现有模块
openset_trainer.py              # 开集训练器
openset_losses.py               # 开集损失函数
openset_methods.py              # 开集检测方法
openset_eval.py                 # 开集评估
stage2.py                       # Stage-2工具函数
```

## 引用

如果使用了本实现，请引用相关论文：

```bibtex
@inproceedings{kang2019decoupling,
  title={Decoupling representation and classifier for long-tailed recognition},
  author={Kang, Bingyi and Xie, Saining and Rohrbach, Marcus and Yan, Zhicheng and Gordo, Albert and Feng, Jiashi and Kalantidis, Yannis},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{bendale2016towards,
  title={Towards open set deep networks},
  author={Bendale, Abhijit and Boult, Terrance E},
  booktitle={CVPR},
  year={2016}
}
```

## 许可证

本实现遵循项目的许可证。
