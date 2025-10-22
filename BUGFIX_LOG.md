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

## 2025-01 - Parameter Passing TypeError

### Issue
修复导入错误后，运行时遇到新错误：

```
TypeError: ADSBSignalDataset.__init__() got an unexpected keyword argument 'alpha_start'
```

### Root Cause
采样器参数（`alpha_start`, `alpha_end`, `alpha`）通过 `**kwargs` 被错误传递给了 `ADSBSignalDataset` 构造函数。

`ADSBSignalDataset` 只接受以下参数：
- `path`, `split`, `data_key`, `label_key`
- `indices`, `in_memory`, `target_length`
- `transforms`, `normalize`, `seed`

但采样器参数（用于 `make_sampler`）也被包含在 `**kwargs` 中。

### Solution
在提交 `5bfc1a9` 中分离了采样器参数和数据集参数：

1. **明确提取采样器参数**:
   ```python
   def create_longtail_openset_dataloaders(
       # ... other params
       alpha: float = 0.5,              # 明确声明
       alpha_start: float = 0.5,        # 明确声明
       alpha_end: float = 0.0,          # 明确声明
       **kwargs,                        # 仅用于数据集参数
   ):
       # 创建采样器参数字典
       sampler_params = {
           'seed': seed,
           'alpha': alpha,
           'alpha_start': alpha_start,
           'alpha_end': alpha_end,
       }
   ```

2. **分离数据集参数**:
   ```python
   # 只提取数据集支持的参数
   dataset_params = {
       'split': kwargs.get('split', None),
       'data_key': kwargs.get('data_key', None),
       'label_key': kwargs.get('label_key', None),
       'in_memory': kwargs.get('in_memory', False),
   }
   # 移除 None 值
   dataset_params = {k: v for k, v in dataset_params.items() if v is not None}
   ```

3. **正确传递参数**:
   ```python
   # 数据集只接收数据集参数
   full_dataset = ADSBSignalDataset(
       path=data_path,
       target_length=target_length,
       transforms=None,
       normalize=False,
       seed=seed,
       **dataset_params,  # ✓ 仅数据集参数
   )

   # 采样器只接收采样器参数
   sampler = make_sampler(
       labels=labels,
       method=sampling_strategy,
       **sampler_params,  # ✓ 仅采样器参数
   )
   ```

### Key Lesson
使用 `**kwargs` 时要小心参数污染：
- ❌ 直接将所有 `**kwargs` 传给不同的函数
- ✅ 明确分离不同函数需要的参数
- ✅ 对于关键参数，使用显式参数而非 `**kwargs`

### Files Changed
- `openset_data_utils.py`: 分离采样器和数据集参数

### Status
✅ **已修复** - 提交 `5bfc1a9`

---

## 2025-01 - AttributeError: label_map

### Issue
修复参数传递后，运行时遇到第三个错误：

```
AttributeError: 'OpenSetSplitter' object has no attribute 'label_map'
```

错误发生在创建长尾分布时访问 `splitter.label_map`。

### Root Cause
`OpenSetSplitter.split()` 方法创建了 `label_map`，但只是返回它，没有存储为实例属性。

```python
def split(self):
    # ... 创建 label_map
    label_map = {...}
    return known_classes, unknown_classes, label_map  # 只返回，未存储
```

后续代码尝试访问：
```python
splitter.label_map  # AttributeError! 属性不存在
```

### Solution
在提交 `928a6da` 中，将返回值同时存储为实例属性：

```python
def split(self):
    # ... 创建 label_map
    label_map = {int(old): int(new) for new, old in enumerate(known_classes_sorted)}

    # ✅ 存储为实例属性
    self.known_classes = known_classes_sorted
    self.unknown_classes = sorted(unknown_classes)
    self.label_map = label_map

    return known_classes_sorted, sorted(unknown_classes), label_map
```

同时修复了 info 字典中的冗余调用：
```python
# 修改前 ❌
"label_map": splitter.split()[2],  # 重复调用 split()

# 修改后 ✅
"label_map": splitter.label_map,   # 直接使用存储的属性
```

### Benefits
1. **修复错误**：代码可以正常访问 `label_map`
2. **避免重复计算**：不需要重新调用 `split()`
3. **更好的封装**：split 结果作为对象状态保存

### Files Changed
- `openset_data_utils.py`: 在 `split()` 中存储实例属性

### Status
✅ **已修复** - 提交 `928a6da`

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
