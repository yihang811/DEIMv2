# DEIMv2 小目标检测头改进方案

## 概述

针对无人机拍摄图像中的小目标检测问题，本方案对DEIMv2的检测头进行了五个方面的增强。所有改进均保持与预训练权重的兼容性，新增模块使用Kaiming初始化。

---

## 五个改进方案

### 方案一：小目标感知查询选择 (Small Object Aware Query Selection)

**文件**: `engine/deim/small_object_head.py` - `SmallObjectAwareQuerySelection`

**问题**: 原始查询选择基于分类分数最大值，小目标特征响应弱，难以被选中。

**改进**:
- 根据anchor面积识别小目标（阈值可配置，默认0.1）
- 为小目标特征添加增强投影
- 预测尺度感知权重，提升小目标的查询选择分数
- 公式: `object_aware_scores = cls_scores * (1.0 + scale_weights * small_object_mask)`

**配置**:
```yaml
DEIMTransformer:
  use_small_object_head: True
  small_object_threshold: 0.1
```

---

### 方案二：多尺度采样点优化 (Multi-Scale Sampling Points)

**文件**: `engine/deim/deim_decoder.py` - `DEIMTransformer.__init__`

**问题**: 所有特征层使用相同数量的采样点（4个），P2层（stride=4）需要更多采样点捕获细节。

**改进**:
- P2层采样点增加到8个
- P3/P4/P5层保持4个采样点
- 配置: `num_points_per_scale: [8, 4, 4, 4]`

**配置**:
```yaml
DEIMTransformer:
  num_points: [8, 4, 4, 4]
  num_points_per_scale: [8, 4, 4, 4]
```

---

### 方案三：尺度自适应回归头 (Scale-Adaptive Regression Head)

**文件**: `engine/deim/small_object_head.py` - `ScaleAdaptiveRegHead`

**问题**: 所有层使用相同的reg_scale（默认4.0），小目标需要更精细的回归。

**改进**:
- 为每个尺度配置独立的回归范围
- P2: 2.0 (小目标精细回归)
- P3: 4.0 (中等目标)
- P4: 6.0 (大目标)
- P5: 8.0 (超大目标)
- 回归参数可学习

**配置**:
```yaml
DEIMTransformer:
  use_scale_adaptive_reg: True
  reg_scales_per_level: [2.0, 4.0, 6.0, 8.0]
```

---

### 方案四：小目标专用检测分支 (Small Object Detection Branch)

**文件**: `engine/deim/small_object_head.py` - `SmallObjectDetectionBranch`

**问题**: 单一检测头难以同时处理大小目标。

**改进**:
- 特征增强层（带Dropout）
- 专用分类头
- 专用回归头
- 目标置信度预测（objectness）
- 可与主检测头并行工作

**结构**:
```
Input Features [bs, num_queries, hidden_dim]
    ↓
Feature Enhancement (Linear + ReLU + Dropout)
    ↓
├─→ Classification Head → cls_logits [bs, num_queries, num_classes]
├─→ Regression Head → reg_corners [bs, num_queries, 4*(reg_max+1)]
└─→ Objectness Head → objectness [bs, num_queries, 1]
```

**配置**:
```yaml
DEIMTransformer:
  use_small_object_head: True  # 同时启用方案一和四
```

---

### 方案五：IoU感知分类优化 (IoU-Aware Classification)

**文件**: `engine/deim/small_object_head.py` - `IoUAwareClassification`, `EnhancedLQE`

**问题**: 分类分数与定位质量不匹配，小目标更容易出现高分类分但低IoU。

**改进**:
- **IoU预测头**: 基于decoder特征和边界框分布预测IoU
- **质量估计头**: 基于分布集中度评估定位质量
- **增强LQE**: 融合IoU预测、质量估计和原始LQE
- 公式: `refined_scores = cls_scores * pred_iou * quality`

**结构**:
```
pred_corners [bs, num_queries, 4*(reg_max+1)]
    ↓
Distribution Statistics (top-k probabilities)
    ↓
├─→ Quality Head → quality score
└─→ IoU Head (w/ decoder features) → IoU score
    ↓
Refined Scores = cls_scores * IoU * quality
```

**配置**:
```yaml
DEIMTransformer:
  use_enhanced_lqe: True
```

---

## 配置文件更新

完整配置示例 (`configs/deimv2/deimv2_dinov3_x_small_object.yml`):

```yaml
DEIMTransformer:
  num_layers: 6
  eval_idx: -1
  num_levels: 4
  feat_channels: [256, 256, 256, 256]
  feat_strides: [4, 8, 16, 32]
  hidden_dim: 256
  dim_feedforward: 2048
  
  # 方案二：多尺度采样点优化
  num_points: [8, 4, 4, 4]
  num_points_per_scale: [8, 4, 4, 4]
  
  cross_attn_method: default
  query_select_method: default
  activation: silu
  mlp_act: silu
  
  # 方案一：小目标感知查询选择
  use_small_object_head: True
  small_object_threshold: 0.1
  
  # 方案三：尺度自适应回归头
  use_scale_adaptive_reg: True
  reg_scales_per_level: [2.0, 4.0, 6.0, 8.0]
  
  # 方案五：IoU感知分类优化
  use_enhanced_lqe: True

optimizer:
  type: AdamW
  params: 
    # ... 原有配置 ...
    - 
      # 小目标检测头模块（新增模块使用较高学习率）
      params: '^(?=.*(?:small_object|scale_adaptive|enhanced_lqe|query_selection)).*$'
      lr: 0.001  # 新模块使用更高学习率
      weight_decay: 0.0001
```

---

## 预训练权重兼容性

### 加载策略

1. **原有权重**: 直接加载，不受影响
2. **新增模块**: 使用Kaiming初始化
3. **部分匹配**: 使用 `strict=False` 加载

### 初始化代码

```python
def _reset_parameters(self, feat_channels):
    # ... 原有初始化 ...
    
    # 初始化小目标检测模块（使用Kaiming初始化，不影响预训练权重）
    if self.use_small_object_head:
        if hasattr(self, 'small_object_query_selection'):
            for m in self.small_object_query_selection.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
        # ... 其他模块类似 ...
```

---

## 训练建议

### 学习率设置

- **骨干网络**: 1e-5 (冻结或微调)
- **编码器/解码器**: 5e-4
- **新增小目标模块**: 1e-3 (更高学习率加速收敛)

### 训练策略

1. **阶段一**: 冻结骨干，训练新增模块 (10-20 epochs)
2. **阶段二**: 解冻骨干，端到端训练
3. **阶段三**: 降低学习率，微调整个模型

### 数据增强

- 使用Mosaic增强（小目标更可能出现在拼接边界）
- RandomZoomOut（模拟小目标场景）
- 避免过度增强导致小目标丢失

---

## 预期效果

| 指标 | 改进前 | 改进后 (预期) |
|------|--------|---------------|
| 小目标AP | 基准 | +5~8% |
| 整体AP | 基准 | +2~4% |
| 大目标AP | 基准 | 保持或略降 |

---

## 文件清单

### 新增文件
- `engine/deim/small_object_head.py` - 小目标检测头模块

### 修改文件
- `engine/deim/deim_decoder.py` - 集成小目标检测增强
- `engine/deim/__init__.py` - 导出新模块
- `configs/deimv2/deimv2_dinov3_x_small_object.yml` - 配置文件

### 测试文件
- `test_small_object_head.py` - 模块测试脚本

---

## 使用说明

### 1. 训练新模型

```bash
python train.py -c configs/deimv2/deimv2_dinov3_x_small_object.yml \
    --resume your_pretrained_weights.pth
```

### 2. 评估模型

```bash
python train.py -c configs/deimv2/deimv2_dinov3_x_small_object.yml \
    --resume your_finetuned_weights.pth \
    --test-only
```

### 3. 仅使用部分改进

可以通过配置文件选择性启用改进方案：

```yaml
# 只启用方案一和方案五
use_small_object_head: True
use_scale_adaptive_reg: False
use_enhanced_lqe: True
```

---

## 注意事项

1. **显存占用**: 新增模块会增加约10-15%显存占用
2. **推理速度**: 略微增加（约5-10%）
3. **小目标定义**: 可通过 `small_object_threshold` 调整
4. **兼容性**: 仅在 `deimv2_dinov3_x_small_object.yml` 配置中启用

---

## 参考文献

- DEIM: DETR with Improved Matching for Fast Convergence
- CBAM: Convolutional Block Attention Module
- D-FINE: Fine-grained Distribution Refinement
