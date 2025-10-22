# Bug Fix Log

## 2025-01 - Import Error Fix

### Issue
运行 `python demo_openset.py --config config_openset.yaml` 时遇到导入错误：

```
ImportError: cannot import name 'SignalDataset' from 'data_utils'
Did you mean: 'ADSBSignalDataset'?
```

### Root Cause
`openset_data_utils.py` 中假设的类名和函数名与实际的 `data_utils.py` 不匹配：

| 假设的名称 | 实际名称 |
|-----------|---------|
| `SignalDataset` | `ADSBSignalDataset` |
| `get_sampler` | `make_sampler` |
| `RandomNoise` | `RandomGaussianNoise` |

同时参数名也不匹配：
- `data_path` → `path`
- `transform` → `transforms`
- `strategy` → `method`

### Solution
已在提交 `4e1ed82` 中修复所有兼容性问题：

1. **类名修正**:
   ```python
   # 修改前
   from data_utils import SignalDataset, get_sampler, RandomNoise

   # 修改后
   from data_utils import ADSBSignalDataset, make_sampler, RandomGaussianNoise
   ```

2. **数据集初始化修正**:
   ```python
   # 修改前
   full_dataset = SignalDataset(
       data_path=data_path,
       transform=None,
   )

   # 修改后
   full_dataset = ADSBSignalDataset(
       path=data_path,
       transforms=None,
   )
   ```

3. **采样器创建修正**:
   ```python
   # 修改前
   sampler = get_sampler(
       labels=labels,
       num_classes=num_classes,
       strategy="progressive_power",
   )

   # 修改后
   sampler = make_sampler(
       labels=labels,
       method="progressive_power",
   )
   ```

4. **变换应用修正**:
   ```python
   # 修改前
   dataset.transform = transform_train

   # 修改后
   dataset.transforms = transform_train
   ```

### Verification
修复后，系统应该能正常运行：

```bash
python demo_openset.py --config config_openset.yaml
```

### Files Changed
- `openset_data_utils.py`: 修复所有API兼容性问题

### Status
✅ **已修复** - 提交 `4e1ed82`

---

## Future Prevention

为避免类似问题，建议：

1. **在编写新模块前先查看实际API**:
   ```bash
   # 查看可用的类和函数
   grep "^class" data_utils.py
   grep "^def" data_utils.py
   ```

2. **使用类型提示和IDE自动补全**

3. **编写基础集成测试**:
   ```python
   def test_import():
       from data_utils import ADSBSignalDataset, make_sampler
       # 测试基本使用
   ```

4. **查看函数签名**:
   ```python
   import inspect
   print(inspect.signature(make_sampler))
   ```

---

**最后更新**: 2025-01
**修复者**: Claude Code
