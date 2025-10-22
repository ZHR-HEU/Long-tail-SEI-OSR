# 基于扩散模型的开集识别 - 技术详解

## 📚 目录
- [核心思想](#核心思想)
- [与传统方法的区别](#与传统方法的区别)
- [数学原理](#数学原理)
- [实现细节](#实现细节)
- [为什么有效](#为什么有效)
- [代码示例](#代码示例)

---

## 核心思想

### 基本原理

**传统扩散模型**（用于生成）:
```
目的：生成新的数据样本
过程：噪声 → 逐步去噪 → 新样本
```

**本系统的创新应用**（用于异常检测）:
```
目的：检测未知类别（开集识别）
过程：特征 → 加噪 → 去噪 → 重构 → 计算误差 → 判断是否异常
```

### 关键创新点

1. **在特征空间操作**，而非原始数据空间
2. **重构误差作为异常分数**
3. **类条件建模**，学习每个已知类的特征分布

---

## 与传统方法的区别

### 对比表

| 维度 | 传统扩散模型 | 本系统的扩散模型 |
|------|-------------|-----------------|
| **输入** | 原始数据（图像、音频等） | 高级特征向量 |
| **目的** | 生成新样本 | 异常检测 |
| **用途** | 数据增强 | 开集识别 |
| **输出** | 生成的新数据 | 重构误差（异常分数） |
| **训练** | 学习数据分布 | 学习已知类特征分布 |
| **推理** | 采样生成 | 重构+误差计算 |

### 图示对比

```
┌─────────────────────────────────────────────────────────────┐
│              传统扩散模型（数据增强）                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始图像 ────┐                                             │
│              ↓                                             │
│         [扩散过程]                                          │
│         加噪 → 完全噪声                                     │
│              ↓                                             │
│         [去噪过程]                                          │
│         学习去噪 → 重构                                     │
│              ↓                                             │
│         新生成的图像 (用于训练)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           本系统：特征空间扩散（开集检测）                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始数据 → CNN → 特征向量 [256维]                          │
│                      ↓                                      │
│                 [训练阶段]                                   │
│            扩散模型学习已知类特征分布                        │
│                      ↓                                      │
│                 [测试阶段]                                   │
│            测试样本特征 → 加噪 → 去噪 → 重构                │
│                                    ↓                        │
│                          计算重构误差                        │
│                                    ↓                        │
│         ┌──────────────────────────┴──────────────────┐     │
│         ↓                                            ↓     │
│    误差小 (< 阈值)                             误差大 (> 阈值)│
│    已知类样本                                   未知类样本   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 数学原理

### 1. 扩散过程（前向，加噪）

给定特征向量 $\mathbf{f}_0$，在时间步 $t$ 添加高斯噪声：

$$
\mathbf{f}_t = \sqrt{\bar{\alpha}_t} \mathbf{f}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

其中：
- $\mathbf{f}_0$: 原始特征（干净）
- $\mathbf{f}_t$: 加噪后的特征
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$: 累积噪声调度参数

**直观理解**：
- 时间步 $t$ 越大，噪声越多
- $t=0$: 无噪声（原始特征）
- $t=T$: 完全噪声

### 2. 去噪过程（反向，重构）

训练一个神经网络 $\epsilon_\theta(\mathbf{f}_t, t, y)$ 来预测添加的噪声：

$$
L = \mathbb{E}_{t, \mathbf{f}_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\mathbf{f}_t, t, y)\|^2 \right]
$$

其中：
- $y$: 类别标签（类条件）
- $\epsilon_\theta$: 去噪网络（本系统中是多层MLP）

**训练目标**：最小化预测噪声与真实噪声的差异

### 3. 重构误差（异常分数）

对于测试样本特征 $\mathbf{f}$：

1. **加噪**：$\mathbf{f}_t = \text{add\_noise}(\mathbf{f}, t)$
2. **去噪**：$\hat{\mathbf{f}}_0 = \text{denoise}(\mathbf{f}_t, t, y)$（多步迭代）
3. **计算误差**：

$$
\text{Anomaly Score} = \|\mathbf{f} - \hat{\mathbf{f}}_0\|_2
$$

**判断规则**：
- 误差小 → 特征能被模型很好地重构 → 属于已知类
- 误差大 → 特征无法被很好重构 → 属于未知类（开集样本）

### 4. 类条件建模

为什么使用类条件？

$$
\epsilon_\theta(\mathbf{f}_t, t, y) = \text{Denoise}(\mathbf{f}_t, \text{TimeEmbed}(t), \text{ClassEmbed}(y))
$$

**作用**：
- 学习每个类别的特征分布
- 已知类 $y=0,1,...,C-1$: 模型学会重构
- 未知类 $y=?$: 无对应的类嵌入，重构困难

---

## 实现细节

### 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    特征扩散模型架构                           │
└──────────────────────────────────────────────────────────────┘

输入特征 [B, D]  (例如: [32, 256])
     │
     ├─────────────────────────────────────┐
     │                                     │
     ↓                                     ↓
[加噪过程]                          [类别标签 y]
ft = √ᾱt·f0 + √(1-ᾱt)·ε                  │
     │                                     │
     │                                     ↓
     │                              [类嵌入层]
     │                           Embedding(y) → [B, E]
     │                                     │
     │                                     │
     ↓                                     │
[时间步 t]                                 │
     │                                     │
     ↓                                     │
[时间嵌入]                                  │
SinusoidalEmbed(t) → MLP → [B, E]         │
     │                                     │
     └──────────┬──────────────────────────┘
                │
                ↓
         [特征 + 时间嵌入 + 类嵌入]
            Concat → [B, D+2E]
                │
                ↓
          [去噪网络 - MLP]
     ┌──────────────────────┐
     │  Linear(D+2E, 512)   │
     │  LayerNorm + GELU    │
     │  Dropout(0.1)        │
     │                      │
     │  Linear(512, 256)    │
     │  LayerNorm + GELU    │
     │  Dropout(0.1)        │
     │                      │
     │  Linear(256, 512)    │
     │  LayerNorm + GELU    │
     │  Dropout(0.1)        │
     │                      │
     │  Linear(512, D)      │
     └──────────────────────┘
                │
                ↓
        预测的噪声 ε̂ [B, D]

训练损失: MSE(ε, ε̂)
```

### 核心代码解析

#### 1. 前向扩散（加噪）

```python
def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
             noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    添加噪声到特征

    x_start: 原始特征 [B, D]
    t: 时间步 [B]
    noise: 噪声 [B, D]（可选，默认采样）

    返回: 加噪特征 [B, D]
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # 获取时间步t对应的系数
    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]  # [B, 1]
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]

    # 加噪公式: x_t = √ᾱt·x0 + √(1-ᾱt)·ε
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

**关键点**：
- `sqrt_alphas_cumprod[t]`: 控制保留多少原始信号
- 时间步 $t$ 越大，噪声占比越高

#### 2. 噪声预测网络

```python
def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                  y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    预测添加的噪声

    x_t: 加噪特征 [B, D]
    t: 时间步 [B]
    y: 类别标签 [B]

    返回: 预测的噪声 [B, D]
    """
    # 时间嵌入
    t_emb = self.time_mlp(t)  # [B, E]

    # 类嵌入（如果是类条件）
    if self.conditional:
        y_emb = self.class_embed(y)  # [B, E]
        condition = torch.cat([t_emb, y_emb], dim=1)  # [B, 2E]
    else:
        condition = t_emb  # [B, E]

    # 拼接加噪特征和条件
    h = torch.cat([x_t, condition], dim=1)  # [B, D+2E]

    # 通过去噪网络预测噪声
    predicted_noise = self.denoising_net(h)  # [B, D]

    return predicted_noise
```

**关键点**：
- 同时使用时间信息和类别信息
- 网络学习预测在特定时间步、特定类别下添加的噪声

#### 3. 单步去噪

```python
def p_sample(self, x_t: torch.Tensor, t: int,
             y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    单步去噪: 从 x_t 恢复到 x_{t-1}

    x_t: t时刻的特征 [B, D]
    t: 当前时间步（标量）
    y: 类别标签 [B]

    返回: x_{t-1} [B, D]
    """
    # 预测噪声
    predicted_noise = self.predict_noise(x_t, t_tensor, y)

    # 计算均值: μ = (1/√αt) * (x_t - (βt/√(1-ᾱt))·ε̂)
    sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
    betas_t = self.betas[t]
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

    model_mean = sqrt_recip_alphas_t * (
        x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t == 0:
        return model_mean  # 最后一步，不加噪声
    else:
        # 添加后验方差噪声
        posterior_variance_t = self.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
```

**关键点**：
- 逐步去除噪声，从 $x_T$ 恢复到 $x_0$
- 每一步都基于预测的噪声进行校正

#### 4. 完整重构过程

```python
def reconstruct(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                num_steps: int = 100) -> torch.Tensor:
    """
    重构特征

    x: 原始特征 [B, D]
    y: 类别标签 [B]
    num_steps: 使用的去噪步数

    返回: 重构的特征 [B, D]
    """
    # 1. 加噪到中间步
    t_start = min(num_steps, self.timesteps - 1)
    t = torch.full((batch_size,), t_start, device=device)
    x_t = self.q_sample(x, t, noise=torch.randn_like(x))

    # 2. 逐步去噪回到 x_0
    for t in reversed(range(t_start + 1)):
        x_t = self.p_sample(x_t, t, y)

    return x_t  # 重构的特征
```

#### 5. 异常分数计算

```python
def compute_reconstruction_error(self, x: torch.Tensor,
                                 y: Optional[torch.Tensor] = None,
                                 num_steps: int = 100) -> torch.Tensor:
    """
    计算重构误差作为异常分数

    x: 测试特征 [B, D]
    y: 预测的类别（或真实类别用于验证）[B]
    num_steps: 去噪步数

    返回: 异常分数 [B]（越大越可能是未知类）
    """
    # 重构特征
    x_recon = self.reconstruct(x, y, num_steps)

    # 计算L2距离作为异常分数
    error = torch.norm(x - x_recon, dim=1)  # [B]

    return error
```

**使用示例**：
```python
# 训练后的模型
diffusion = FeatureDiffusion(feature_dim=256, num_classes=6)

# 测试样本
features = extract_features(test_data)  # [N, 256]
predictions = classifier(test_data)      # [N] 预测类别

# 计算异常分数
anomaly_scores = diffusion.compute_reconstruction_error(
    features,
    predictions,
    num_steps=100
)

# 开集检测
threshold = 2.5  # 根据验证集确定
is_unknown = anomaly_scores > threshold

print(f"Unknown samples: {is_unknown.sum()}")
```

---

## 为什么有效？

### 1. 理论基础

**已知类样本**：
- 训练时见过，扩散模型学习了其特征分布
- 重构时，模型知道如何去噪
- 重构误差小

**未知类样本**：
- 训练时未见过，特征分布不同
- 重构时，模型无法正确去噪（没学过这种模式）
- 重构误差大

### 2. 直观解释

想象扩散模型是一个"特征修复专家"：

```
训练：学习修复6种已知类型的"损坏"特征
      猫、狗、鸟、鱼、马、牛

测试时：
  - 给它一个"损坏"的猫特征 → 能很好修复（误差小）✓
  - 给它一个"损坏"的狗特征 → 能很好修复（误差小）✓
  - 给它一个"损坏"的蛇特征 → 修不好（误差大）✗
    因为从没见过蛇！
```

### 3. 类条件的作用

```python
# 无类条件（只用时间）
noise_pred = denoise_net(x_t, time_emb)
# 问题：所有类混在一起，界限模糊

# 有类条件（时间+类别）
noise_pred = denoise_net(x_t, time_emb, class_emb)
# 优势：每个类别有专门的"修复方案"
#       类别0用方案0，类别1用方案1...
#       未知类？没有对应方案，修不好！
```

### 4. 与其他方法的互补

| 方法 | 判断依据 | 优势 | 局限 |
|------|---------|------|------|
| **Softmax** | 最大概率 | 简单快速 | 容易过度自信 |
| **ODIN** | 温度缩放 | 改善校准 | 仍基于logits |
| **OpenMax** | 极值理论 | 理论完备 | 需要拟合分布 |
| **扩散模型** | 重构质量 | 深度特征分析 | 计算稍慢 |

**组合使用**最佳：
```python
# 综合多个信号
final_score = 0.4 * diffusion_score +
              0.3 * openmax_score +
              0.3 * energy_score
```

---

## 代码示例

### 完整使用流程

```python
import torch
from diffusion_models import create_feature_diffusion
from models import ConvNetADSB

# ============================================
# 1. 创建模型
# ============================================

# 特征提取器（CNN）
backbone = ConvNetADSB(num_classes=6)

# 扩散模型
diffusion = create_feature_diffusion(
    feature_dim=256,          # 特征维度
    num_classes=6,            # 已知类别数
    timesteps=1000,           # 扩散步数
    hidden_dims=[512, 256, 512],  # 去噪网络结构
    beta_schedule="cosine",   # 噪声调度
    conditional=True,         # 使用类条件
    enhanced=True             # 增强版（多时间步）
)

# ============================================
# 2. 训练阶段
# ============================================

for epoch in range(num_epochs):
    for batch in train_loader:
        x, y = batch  # x: [B, C, T], y: [B]

        # 提取特征
        features = backbone.extract_features(x)  # [B, 256]

        # 计算扩散损失
        diffusion_loss, _ = diffusion(features, y)

        # 计算分类损失
        logits = backbone.classifier(features)
        cls_loss = F.cross_entropy(logits, y)

        # 联合优化
        total_loss = cls_loss + 0.1 * diffusion_loss
        total_loss.backward()
        optimizer.step()

# ============================================
# 3. 测试阶段 - 开集检测
# ============================================

backbone.eval()
diffusion.eval()

test_features = []
test_labels = []
anomaly_scores = []

for batch in test_loader:
    x, y_true = batch  # y_true可能包含-1（未知类）

    # 提取特征
    features = backbone.extract_features(x)

    # 预测类别
    logits = backbone.classifier(features)
    y_pred = torch.argmax(logits, dim=1)

    # 计算异常分数
    scores = diffusion.compute_anomaly_score(
        features,
        y_pred,  # 使用预测类别
        method="reconstruction",
        num_steps=100
    )

    anomaly_scores.append(scores.cpu())
    test_labels.append(y_true.cpu())

# 转换为numpy
anomaly_scores = torch.cat(anomaly_scores).numpy()
test_labels = torch.cat(test_labels).numpy()

# ============================================
# 4. 设置阈值（基于验证集）
# ============================================

# 已知类样本的异常分数
known_scores = anomaly_scores[test_labels >= 0]

# 设置阈值为95百分位
threshold = np.percentile(known_scores, 95)
print(f"Threshold: {threshold:.3f}")

# ============================================
# 5. 开集识别
# ============================================

# 判断是否为未知类
is_unknown = anomaly_scores > threshold

# 评估
from sklearn.metrics import roc_auc_score

# 真实标签：1=未知，0=已知
true_unknown = (test_labels < 0).astype(int)

# 计算AUROC
auroc = roc_auc_score(true_unknown, anomaly_scores)
print(f"AUROC: {auroc:.4f}")

# 统计
n_unknown_true = (test_labels < 0).sum()
n_unknown_pred = is_unknown.sum()
print(f"True unknown: {n_unknown_true}")
print(f"Predicted unknown: {n_unknown_pred}")
```

### 可视化重构过程

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 选择一些测试样本
n_samples = 10
features_test = test_features[:n_samples]  # [10, 256]
labels_test = test_labels[:n_samples]

# 重构
features_recon = diffusion.diffusion.reconstruct(
    features_test,
    labels_test,
    num_steps=100
)

# 降维到2D可视化
pca = PCA(n_components=2)
features_all = torch.cat([features_test, features_recon], dim=0)
features_2d = pca.fit_transform(features_all.cpu().numpy())

# 分离原始和重构
original_2d = features_2d[:n_samples]
recon_2d = features_2d[n_samples:]

# 绘图
plt.figure(figsize=(10, 8))
for i in range(n_samples):
    label = labels_test[i].item()
    color = 'red' if label < 0 else 'blue'

    # 原始特征
    plt.scatter(original_2d[i, 0], original_2d[i, 1],
                c=color, marker='o', s=100, alpha=0.6,
                label='Original' if i == 0 else '')

    # 重构特征
    plt.scatter(recon_2d[i, 0], recon_2d[i, 1],
                c=color, marker='x', s=100, alpha=0.6,
                label='Reconstructed' if i == 0 else '')

    # 连线显示重构误差
    plt.plot([original_2d[i, 0], recon_2d[i, 0]],
             [original_2d[i, 1], recon_2d[i, 1]],
             'k--', alpha=0.3)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Feature Reconstruction Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('diffusion_reconstruction.png', dpi=300)
plt.show()
```

---

## 高级技巧

### 1. 多时间步评估

```python
# 在多个时间步计算重构误差，取平均
def robust_anomaly_score(diffusion, features, labels):
    timesteps_to_test = [50, 100, 150, 200]
    scores = []

    for num_steps in timesteps_to_test:
        score = diffusion.compute_reconstruction_error(
            features, labels, num_steps
        )
        scores.append(score)

    # 平均多个时间步的结果
    return torch.stack(scores).mean(dim=0)
```

### 2. 似然估计（备选方法）

```python
# 使用负对数似然作为异常分数
likelihood_scores = diffusion.compute_likelihood(
    features, labels, num_timesteps=10
)

# 结合重构误差和似然
combined_score = 0.6 * reconstruction_error + 0.4 * likelihood_scores
```

### 3. 集成多个检测器

```python
# 扩散模型 + OpenMax + Energy
from openset_methods import OpenMax, EnergyBasedOOD

diffusion_scores = diffusion.compute_anomaly_score(features, labels)
openmax_scores, _ = openmax.predict(features, logits)
energy_scores, _ = energy.predict(logits)

# 加权集成
final_scores = (0.4 * diffusion_scores +
                0.3 * openmax_scores +
                0.3 * energy_scores)
```

---

## 总结

### 关键要点

1. **在特征空间操作**：高效且有效
2. **重构误差判断**：简单直观的异常度量
3. **类条件建模**：学习每个类的特征分布
4. **训练简单**：只需MSE损失
5. **推理灵活**：可调整去噪步数权衡速度/精度

### 优势

✅ 深度特征分析，而非表面logits
✅ 自然处理分布外样本
✅ 可解释性强（重构质量）
✅ 与其他方法互补
✅ 适用于长尾场景

### 局限

❌ 计算比MSP/Energy慢（需要多步去噪）
❌ 需要额外训练扩散模型
❌ 超参数需要调优（timesteps, num_steps等）

### 适用场景

最适合：
- 特征分布复杂的数据（如RF信号、医学图像）
- 对精度要求高于速度的场景
- 已知类和未知类特征差异明显

不太适合：
- 实时性要求极高的场景
- 计算资源受限的环境
- 已知类和未知类特征非常相似

---

## 参考文献

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **Score-based Models**: Song & Ermon, "Generative Modeling by Estimating Gradients", NeurIPS 2019
3. **Diffusion for Anomaly Detection**: Wyatt et al., "AnoDDPM: Anomaly Detection with Denoising Diffusion", MICCAI 2022
4. **Open-Set Recognition**: Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016

---

**文档版本**: 1.0
**最后更新**: 2025-01
