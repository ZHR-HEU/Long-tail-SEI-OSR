# Fix configuration issues and add comprehensive documentation

## 🎯 Summary

This PR fixes critical configuration issues in the two-stage open-set training pipeline and adds comprehensive documentation.

## 🐛 Fixed Issues

### 1. **Data Path Configuration**
- ✅ Unified data paths across all config files
- Updated `config_openset.yaml` and `config_openset_twostage.yaml` to use correct paths
- All configs now use: `/home/dell/md3/zhahaoran/data/ADS-B_*.mat`

### 2. **norm_kind ValueError**
- ❌ **Before**: `norm_kind: "batch_norm"` (not supported)
- ✅ **After**: `norm_kind: "auto"` (supported)
- Model supports: `"auto"`, `"bn"`, `"gn"`, `"ln"`

### 3. **Loss Configuration Format Error**
- ❌ **Before**: Simple format with `name: "Focal"`
- ✅ **After**: Complete format with `loss_type`, `use_diffusion`, `use_contrastive` flags
- Fixed ValueError: diffusion_model must be provided when use_diffusion=True

## 📚 Documentation Added

### New Documentation Files

1. **RUN_GUIDE.md** (English)
   - Detailed setup and usage guide
   - Comprehensive troubleshooting (10+ common issues)
   - Parameter tuning recommendations
   - Multiple scenario configurations

2. **快速运行指南.md** (Chinese)
   - Quick 3-step getting started
   - Common errors quick reference table
   - Training time estimates
   - Essential parameter adjustments

3. **CONFIG_GUIDE.md**
   - Comparison of 3 config files
   - Decision tree for config selection
   - Complete parameter reference
   - Performance comparison table

## 🔧 Technical Changes

### Config Files Modified

**config_openset_twostage.yaml**:
```yaml
# Fixed model configuration
model:
  norm_kind: "auto"  # Was: "batch_norm"

# Fixed Stage-1 loss configuration
stage1:
  loss:
    loss_type: "focal"
    use_diffusion: false
    use_contrastive: false
    use_entropy: false
    gamma: 2.0
    alpha: 0.25

# Fixed Stage-2 loss configuration
stage2:
  loss:
    loss_type: "balanced_softmax"
    use_diffusion: false
    use_contrastive: true
    lambda_contrastive: 0.1
```

**config_openset.yaml**:
- Updated data paths to match main config

## ✅ Testing

The fixes resolve the following runtime errors:
- ✅ `ValueError: Unknown norm kind: batch_norm`
- ✅ `ValueError: diffusion_model must be provided when use_diffusion=True`
- ✅ Data file path issues

## 📖 Usage

After merging, users can:
```bash
# Quick start (just 1 command!)
python train_openset_twostage.py

# Expected behavior: no configuration errors
```

## 🎯 Impact

- **Before**: Training fails immediately with config errors
- **After**: Training runs successfully with correct configuration
- **Documentation**: Complete guides for new users

## 📋 Checklist

- [x] Fixed all configuration errors
- [x] Added comprehensive documentation
- [x] Updated troubleshooting guides
- [x] Provided example configurations
- [x] No Python code changes (config-only fixes)
- [x] All changes tested and working

## 🤖 Notes

This PR contains only configuration file and documentation updates. No Python code was modified, ensuring stability and safety.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
