# Long-tail SEI Open-Set Recognition 使用指南

本项目提供三个针对 ADS-B 信号长尾分类与开放集识别的训练流程，每个流程均由独立的配置文件与入口脚本驱动：

1. `config.yaml` 搭配 `main.py`，用于两阶段长尾分类（基础训练 + 分类器再训练）。
2. `config_openset.yaml` 搭配 `demo_openset.py`，用于单阶段长尾开放集识别演示。
3. `config_openset_twostage.yaml` 搭配 `train_openset_twostage.py`，用于两阶段开放集训练流程。

下文详细说明数据放置方式、关键配置段落以及每个脚本的运行方法。

## 数据准备

- 所有配置均默认读取 ADS-B 数据集，常用文件包括 `ADS-B_Train_100X360-2_5-10-15-20dB.mat` 与 `ADS-B_test_100X40_5-10-15-20dB.mat`。
- 请在三个配置文件中的 `data.path_train`、`data.path_test`（以及需要时的 `data.path_val`）字段填入实际路径，可使用绝对或相对路径。
- 如果没有单独的验证集，`main.py` 在 `data.path_val` 为空时会自动从训练集切分验证子集。

## 使用 `config.yaml` 与 `main.py`

`main.py` 利用 Hydra 管理配置，用于先训练基础模型，再执行分类器再训练（Classifier Re-Training, CRT）。

### 关键配置说明

- `data`：批大小、加载线程数、采样策略（如 `class_balance`）、信号截断长度等。
- `model`：`ConvNetADSB` 的层数、通道数、投影维度等超参数。
- `loss` / `sampling`：第一阶段的损失函数（如 `balanced_softmax`）与重采样方案。
- `training`：优化器、学习率、调度策略、混合精度 (`amp`) 设置。
- `stage2`：控制是否启用 CRT、进阶采样 (`progressive_sampler`)、代价敏感损失等细节。
- `gpus` 字段可填写形如 `"0,1"` 的设备列表；设为 `"cpu"` 可强制使用 CPU。

### 运行方式

直接执行脚本会根据 `config.yaml` 中的默认参数训练，并把所有输出写入 `experiments/<exp_name>_<timestamp>/` 目录。

```bash
python main.py
```

需要临时覆写参数时，可借助 Hydra 的命令行语法，例如修改实验名、批大小并跳过第二阶段：

```bash
python main.py exp_name=my_test_run data.batch_size=512 stage2.enabled=false
```

如果希望固定输出目录，可追加 `hydra.run.dir=.`，让结果直接保存到当前工作路径。

## 使用 `config_openset.yaml` 与 `demo_openset.py`

该脚本实现单阶段开放集识别示例，支持多种开放集检测器（OpenMax、ODIN、Energy、Mahalanobis、MSP 等）以及扩散式正则化。

### 关键配置说明

- `data`：已知类别列表、类间不平衡倍率、逐阶段采样日程、数据增强设置。
- `diffusion`：是否启用特征扩散模型、训练轮数、温度等超参数。
- `openset`：选择检测器类型（默认 `openmax`）并配置尾部大小、温度或阈值。
- `loss`：组合分类损失、对比损失、对象球正则项及扩散损失的系数。
- `training` / `evaluation`：学习率、调度器、保存频率，以及指标（AUROC、OSCR 等）记录方式。
- `visualization`：开启后会在 `./visualizations_openset` 中生成曲线和特征可视化。

### 运行方式

命令行参数 `--config` 可显式指定配置文件位置，模型权重与评估结果默认写入 `training.checkpoint_dir` 定义的目录（默认为 `./checkpoints_openset/`）。

```bash
python demo_openset.py --config config_openset.yaml
```

若需切换检测器或调整阈值，可通过命令行覆写，如将检测器改为 Energy 并调整温度：

```bash
python demo_openset.py --config config_openset.yaml openset.method=energy openset.energy.temperature=2.0
```

## 使用 `config_openset_twostage.yaml` 与 `train_openset_twostage.py`

该流程将表示学习与开放集校准拆分为两个阶段，既能复用长尾分类技巧，又能在第二阶段针对开放集分数进行细化。

### 阶段划分

1. **Stage 1**：在已知类别上训练主干网络，可结合 `stage1.diffusion.enabled=true` 引入扩散增强。
2. **Stage 2**：冻结或微调主干，根据 `stage2.mode` 选择 CRT 或微调策略，同时拟合开放集检测器。

### 关键配置说明

- `stage1.optimizer`、`stage1.scheduler`：控制第一阶段的学习率和调度策略。
- `stage1.loss`：支持焦点损失、重加权等多种长尾友好损失。
- `stage2.classifier`：分别指定分类头与主干的学习率、动量、是否只更新分类头。
- `stage2.openset`：配置检测器类型、拟合频率、尾部大小等；可单独启停某些检测器。
- `evaluation`：控制是否导出 logits、特征缓存以及最终指标汇总文件名（默认 `final_results.txt`）。
- `checkpoints`：`stage1` 与 `stage2` 子字段决定权重保存目录，默认分别为 `./checkpoints/stage1` 与 `./checkpoints/stage2`。

### 运行方式

执行以下命令将按默认配置完成两个阶段，并在日志中打印关键指标。

```bash
python train_openset_twostage.py --config config_openset_twostage.yaml
```

可根据实验需要覆盖部分参数，例如将第二阶段切换为微调并降低分类头学习率：

```bash
python train_openset_twostage.py --config config_openset_twostage.yaml stage2.mode=finetune stage2.classifier.head_lr=0.0005
```

## 日志与结果定位

- 所有脚本都会在各自的日志目录生成 `train.log`（或同等命名）以及 `final_results.txt`，方便快速查看最终指标。
- `main.py` 与 `train_openset_twostage.py` 默认启用 TensorBoard 事件文件，可在对应输出目录运行 `tensorboard --logdir <path>` 进行可视化。
- 若需要复现实验，建议将自定义配置文件保存到 `configs/` 子目录，并在命令行中引用，例如 `python main.py --config-name my_experiment`。

## 常见问题

- **路径报错**：确认 YAML 中的路径与数据文件名完全一致；Hydra 会切换工作目录到实验输出路径，建议使用绝对路径或在命令中追加 `hydra.run.dir=.`。
- **显存不足**：可减小 `data.batch_size` 或启用 `training.use_amp=true`（若脚本中提供该选项）。
- **开放集分数异常**：检查 `openset` 配置中阈值、尾部大小等是否适合当前数据规模；必要时降低尾部大小或增加拟合频率。
- **继续训练/评估**：多数脚本支持从断点恢复或仅运行评估，可在配置中设置 `training.resume_from` 或 `evaluation.only=true`。

以上内容覆盖了三个核心配置文件的使用流程，可根据实际需求复制并修改 YAML 文件以扩展更多实验。
