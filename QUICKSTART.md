# 快速开始指南 - 长尾开集识别系统

## 🚀 5分钟快速上手

### 步骤 1: 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n openset python=3.9
conda activate openset

# 安装依赖
pip install torch torchvision  # 或使用CUDA版本
pip install numpy scipy scikit-learn pyyaml tqdm h5py matplotlib
```

### 步骤 2: 验证安装

```bash
python test_openset_system.py
```

如果看到 "✓ All core components are working correctly!"，说明系统正常运行。

### 步骤 3: 准备数据

准备您的数据文件（支持 .mat、.h5、.npy 格式）：

```python
# 数据格式示例
import numpy as np
import scipy.io as sio

# 创建示例数据
data = {
    'X': np.random.randn(1000, 2, 4800),  # [样本数, 通道数, 序列长度]
    'Y': np.random.randint(0, 8, 1000),    # [样本数] 标签
}

# 保存为MAT文件
sio.savemat('train_data.mat', data)
```

### 步骤 4: 配置参数

编辑 `config_openset.yaml`：

```yaml
data:
  path_train: "train_data.mat"  # 您的训练数据路径
  num_known_classes: 6          # 已知类别数（总类别数的一部分）
  imbalance_ratio: 100.0        # 长尾比例

diffusion:
  enabled: true                 # 启用扩散模型

openset:
  detector_type: "openmax"      # 开集检测方法
```

### 步骤 5: 开始训练

```bash
python demo_openset.py --config config_openset.yaml
```

训练过程会显示：
- 每个epoch的训练损失
- 验证集上的各项指标（AUROC, OSCR等）
- 最佳模型保存信息

### 步骤 6: 查看结果

训练完成后，结果保存在：
```
checkpoints_openset/
├── best_model.pth          # 最佳模型权重
├── final_results.txt       # 详细测试结果
└── checkpoint_epoch_*.pth  # 定期检查点
```

## 📊 预期输出示例

```
================================================================================
                        Test Set Results
================================================================================

Dataset Info:
  Total test samples: 200
  Unknown samples: 50
  Known classes: 6

Closed-Set Metrics (Known Classes Only):
  Accuracy: 0.8533

Open-Set Detection Metrics:
  AUROC: 0.8842
  AUPR: 0.8156
  FPR95: 0.1234

Open-Set Classification Metrics:
  OSCR: 0.7891
  F1-Score: 0.8245
  Overall Accuracy: 0.8100

Long-Tail Analysis:
  Many-shot Acc: 0.9100
  Medium-shot Acc: 0.8200
  Few-shot Acc: 0.6500

================================================================================
```

## 🎯 关键概念速览

### 什么是长尾开集识别？

**长尾问题**：训练数据中类别分布不平衡
```
头部类别: ████████████████████ (很多样本)
中部类别: ██████████ (中等样本)
尾部类别: ███ (很少样本)
```

**开集问题**：测试时出现训练时未见过的类别
```
训练: 类别 0, 1, 2, 3, 4, 5 (已知)
测试: 类别 0, 1, 2, 3, 4, 5, 6, 7 (包含未知类6和7)
      ↑已知类           ↑未知类
```

### 扩散模型的创新用法

**传统用法** (数据增强):
```
原始数据 → 加噪 → 去噪 → 生成新数据 → 用于训练
```

**本系统的创新用法** (异常检测):
```
特征向量 → 加噪 → 去噪 → 重构特征
          ↓
    重构误差大 → 可能是未知类 (开集样本)
    重构误差小 → 可能是已知类
```

## 🔧 常见调优

### 提升开集检测性能

```yaml
diffusion:
  timesteps: 1000           # 增加时间步
  lambda_diffusion: 0.2     # 增大扩散损失权重

openset:
  detector_type: "openmax"  # 使用OpenMax而非MSP
```

### 提升尾部类别性能

```yaml
loss:
  loss_type: "balanced_softmax"  # 使用平衡softmax
  use_contrastive: true          # 启用对比学习
  lambda_contrastive: 0.2        # 增大对比损失权重

data:
  sampling_strategy: "progressive_power"  # 渐进式采样
```

### 加速训练

```yaml
data:
  batch_size: 512           # 增大批大小

diffusion:
  timesteps: 500            # 减少时间步
  hidden_dims: [256, 128, 256]  # 减小网络

training:
  epochs: 100               # 减少训练轮数
```

## 📈 实验建议

### 基线对比实验

1. **不使用扩散模型**
```yaml
diffusion:
  enabled: false
```

2. **不同开集检测方法**
```bash
# 测试所有方法
for method in msp odin energy openmax mahalanobis; do
  # 修改config中的detector_type
  python demo_openset.py --config config_${method}.yaml
done
```

3. **不同长尾处理方法**
```yaml
loss:
  loss_type: "ce"                    # 基线
  loss_type: "focal"                 # Focal Loss
  loss_type: "balanced_softmax"      # Balanced Softmax
  loss_type: "ldam"                  # LDAM
```

### 消融实验

测试每个组件的贡献：

| 配置 | 扩散 | 对比 | 采样 | 预期AUROC |
|------|------|------|------|-----------|
| 基线 | ✗ | ✗ | 无 | ~0.75 |
| +扩散 | ✓ | ✗ | 无 | ~0.82 |
| +对比 | ✗ | ✓ | 无 | ~0.78 |
| +采样 | ✗ | ✗ | ✓ | ~0.77 |
| 完整 | ✓ | ✓ | ✓ | ~0.88 |

## 🐛 故障排除

### 问题1: CUDA out of memory

**解决方案**:
```yaml
data:
  batch_size: 64  # 减小批大小
diffusion:
  hidden_dims: [256, 128, 256]  # 减小网络
```

### 问题2: 训练不收敛

**解决方案**:
```yaml
training:
  lr: 5e-4  # 降低学习率
  scheduler: "cosine"  # 使用cosine调度

loss:
  lambda_diffusion: 0.05  # 降低扩散损失权重
```

### 问题3: 开集检测效果差

**解决方案**:
1. 增加训练轮数
2. 使用OpenMax而非MSP
3. 增大扩散损失权重
4. 确保已知/未知类别有明显差异

### 问题4: 尾部类别准确率低

**解决方案**:
```yaml
data:
  sampling_strategy: "class_uniform"  # 类均匀采样

loss:
  loss_type: "balanced_softmax"
  use_contrastive: true
  lambda_contrastive: 0.2
```

## 📚 进阶使用

### 自定义模型

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 您的模型定义

    def forward_with_features(self, x):
        """必须实现此方法"""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

# 在demo_openset.py中使用
model = MyCustomModel(num_classes=6)
```

### 自定义损失

```python
from openset_losses import LongTailOpenSetLoss

# 继承并扩展
class MyCustomLoss(LongTailOpenSetLoss):
    def forward(self, logits, labels, features, diffusion_model):
        # 基础损失
        base_loss, loss_dict = super().forward(
            logits, labels, features, diffusion_model
        )

        # 添加自定义损失
        custom_loss = ...  # 您的损失
        total_loss = base_loss + 0.1 * custom_loss

        return total_loss, loss_dict
```

### 可视化结果

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 提取特征
features, labels = trainer.extract_features(test_loader)

# t-SNE可视化
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(features)

plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar()
plt.title('Feature Space Visualization')
plt.savefig('tsne_visualization.png')
```

## 💡 最佳实践

1. **数据准备**
   - 确保数据标准化
   - 检查类别分布
   - 验证已知/未知类别划分合理

2. **模型训练**
   - 从小模型开始测试
   - 使用余弦学习率调度
   - 早停防止过拟合

3. **超参数调优**
   - 先调整基础学习率
   - 再调整各损失权重
   - 最后微调采样策略

4. **评估验证**
   - 关注多个指标（不只是准确率）
   - 分析长尾分组性能
   - 检查混淆矩阵

## 🎓 学习资源

- [详细文档](README_OPENSET.md)
- [模块API参考](CODEBASE_SUMMARY.md)
- [配置参数说明](config_openset.yaml)

## 📞 获取帮助

- 查看 [常见问题](README_OPENSET.md#常见问题)
- 运行测试: `python test_openset_system.py`
- 提Issue到GitHub仓库

---

**祝您实验顺利！** 🎉
