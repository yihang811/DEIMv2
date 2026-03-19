# DEIMv2 小目标检测特征流向详解

本文档详细描述从输入图像到最终检测结果的完整特征流向，包括小目标检测改进后的数据流。

---

## 整体架构概览

```
输入图像 [B, 3, H, W] (例如: [1, 3, 640, 640])
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: 骨干网络 (Backbone - DINOv3STAs)                    │
│ - 提取多尺度特征 (P2, P3, P4, P5)                           │
└─────────────────────────────────────────────────────────────┘
    ↓
特征列表 [P2, P3, P4, P5]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: 编码器 (Encoder - HybridEncoder)                    │
│ - 特征融合与增强                                            │
│ - 小目标特征增强模块 (SmallObjectEnhancementModule)         │
└─────────────────────────────────────────────────────────────┘
    ↓
增强特征 [B, 256, H/4, W/4] 等
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: 解码器 (Decoder - DEIMTransformer)                  │
│ - 查询选择 (小目标感知)                                      │
│ - 多尺度可变形注意力                                         │
│ - 检测头 (分类+回归)                                         │
└─────────────────────────────────────────────────────────────┘
    ↓
检测结果 {'pred_logits': [B, 300, 80], 'pred_boxes': [B, 300, 4]}
```

---

## Stage 1: 骨干网络 (Backbone)

### 输入
- **图像尺寸**: `[B, 3, 640, 640]` (Batch, Channel, Height, Width)
- **示例**: `[1, 3, 640, 640]`

### 1.1 空间先验模块 (Spatial Prior Module)

```
输入图像 [1, 3, 640, 640]
    ↓
┌────────────────────────────────────────┐
│ stem_conv (stride=2)                   │
│ Conv(3→16, 3×3, s=2) + BN + GELU       │
└────────────────────────────────────────┘
    ↓ [1, 16, 320, 320]
    ↓
┌────────────────────────────────────────┐
│ p2_conv (stride=2)                     │
│ Conv(16→32, 3×3, s=2) + BN + GELU      │
└────────────────────────────────────────┘
    ↓ [1, 32, 160, 160]  ← P2原始特征 (stride=4)
    ↓
┌────────────────────────────────────────┐
│ conv2 (stride=2)                       │
│ Conv(32→32, 3×3, s=2) + BN             │
└────────────────────────────────────────┘
    ↓ [1, 32, 80, 80]    ← P3原始特征 (stride=8)
    ↓
┌────────────────────────────────────────┐
│ conv3 (stride=2)                       │
│ GELU + Conv(32→64, 3×3, s=2) + BN      │
└────────────────────────────────────────┘
    ↓ [1, 64, 40, 40]    ← P4原始特征 (stride=16)
    ↓
┌────────────────────────────────────────┐
│ conv4 (stride=2)                       │
│ GELU + Conv(64→64, 3×3, s=2) + BN      │
└────────────────────────────────────────┘
    ↓ [1, 64, 20, 20]    ← P5原始特征 (stride=32)
```

### 1.2 DINOv3 Transformer特征提取

```
输入图像 [1, 3, 640, 640]
    ↓
Patch Embedding (patch_size=16)
    ↓ [1, 1600, 384]  (1600 = 40×40 patches)
    ↓
Transformer Blocks (layers 2, 5, 8, 11)
    ↓
多层级特征提取:
    - Layer 2:  [1, 1600, 384]  → 上采样 → [1, 384, 160, 160] (P2)
    - Layer 5:  [1, 1600, 384]  → 上采样 → [1, 384, 80, 80]   (P3)
    - Layer 8:  [1, 1600, 384]  → 上采样 → [1, 384, 40, 40]   (P4)
    - Layer 11: [1, 1600, 384]  → 上采样 → [1, 384, 20, 20]   (P5)
```

### 1.3 特征融合投影

```
对每个尺度 (P2, P3, P4, P5):
    
DINOv3特征 [1, 384, H, W]
    ↓
空间先验特征 [1, 32/32/64/64, H, W]
    ↓
Concatenate → [1, 384+32/64, H, W]
    ↓
Conv(384+32/64 → 256, 1×1)  # 投影到hidden_dim
    ↓
BatchNorm + 激活

输出:
    - P2: [1, 256, 160, 160]
    - P3: [1, 256, 80, 80]
    - P4: [1, 256, 40, 40]
    - P5: [1, 256, 20, 20]
```

### Stage 1 输出

```python
feat_list = [
    feat_p2,  # [B, 256, 160, 160]  stride=4
    feat_p3,  # [B, 256, 80, 80]    stride=8
    feat_p4,  # [B, 256, 40, 40]    stride=16
    feat_p5,  # [B, 256, 20, 20]    stride=32
]
```

---

## Stage 2: 编码器 (HybridEncoder)

### 2.1 输入投影

```
P2: [1, 256, 160, 160] → Conv(256→256, 1×1) + BN → [1, 256, 160, 160]
P3: [1, 256, 80, 80]   → Conv(256→256, 1×1) + BN → [1, 256, 80, 80]
P4: [1, 256, 40, 40]   → Conv(256→256, 1×1) + BN → [1, 256, 40, 40]
P5: [1, 256, 20, 20]   → Conv(256→256, 1×1) + BN → [1, 256, 20, 20]
```

### 2.2 内部特征处理 (Intra-scale)

```
对每个特征层应用CSPLayer:

输入特征 [B, 256, H, W]
    ↓
┌────────────────────────────────────────┐
│ CSPLayer (3 blocks, expansion=1.0)     │
│ ├─ conv1: Conv(256→256, 1×1)           │
│ ├─ bottlenecks (3× VGGBlock)           │
│ │   └─ Conv(256→256, 3×3) + Conv(256→256, 1×1)│
│ └─ conv2: Conv(256→256, 1×1)           │
│ Output: conv3(conv1_out + conv2_out)   │
└────────────────────────────────────────┘
    ↓ [B, 256, H, W]
```

### 2.3 跨尺度特征融合 (Cross-scale)

```
自上而下 (Top-down) 路径:

P5 [B, 256, 20, 20]
    ↓
Upsample ×2 → [B, 256, 40, 40]
    ↓
Concat with P4 [B, 256, 40, 40] → [B, 512, 40, 40]
    ↓
RepNCSPELAN4 → [B, 256, 40, 40] (新P4)
    ↓
Upsample ×2 → [B, 256, 80, 80]
    ↓
Concat with P3 [B, 256, 80, 80] → [B, 512, 80, 80]
    ↓
RepNCSPELAN4 → [B, 256, 80, 80] (新P3)
    ↓
Upsample ×2 → [B, 256, 160, 160]
    ↓
Concat with P2 [B, 256, 160, 160] → [B, 512, 160, 160]
    ↓
RepNCSPELAN4 → [B, 256, 160, 160] (新P2)

自下而上 (Bottom-up) 路径:

P2 [B, 256, 160, 160]
    ↓
SCDown(s=2) → [B, 256, 80, 80]
    ↓
Concat with P3 → [B, 512, 80, 80]
    ↓
RepNCSPELAN4 → [B, 256, 80, 80] (最终P3)

... (P4, P5类似)
```

### 2.4 小目标特征增强 (SmallObjectEnhancementModule)

```
输入:
    - P2: [B, 256, 160, 160]
    - P3: [B, 256, 80, 80]

┌─────────────────────────────────────────────────────────┐
│ SmallObjectEnhancementModule                            │
│                                                         │
│ P2处理分支:                                              │
│   P2 [B, 256, 160, 160]                                 │
│     ↓                                                   │
│   ├─→ CBAM注意力 (Channel + Spatial)                   │
│   │   ├─ ChannelAttention: AvgPool/MaxPool → FC × 2    │
│   │   └─ SpatialAttention: Avg/Max → Conv(7×7)         │
│   ↓                                                     │
│   ├─→ DepthwiseSeparableConv × 2                       │
│   │   └─ DWConv(3×3) + PWConv(1×1) + BN + SiLU         │
│   ↓                                                     │
│   └─→ Residual Connection                              │
│   P2_enhanced [B, 256, 160, 160]                        │
│                                                         │
│ P3处理分支:                                              │
│   P3 [B, 256, 80, 80]                                   │
│     ↓                                                   │
│   └─→ ContextAggregation (dilations=[1,2,3,6])          │
│       └─ 4并行分支 (不同dilation) → Concat → Conv       │
│   P3_context [B, 256, 80, 80]                           │
│                                                         │
│ 融合:                                                   │
│   P2_enhanced [B, 256, 160, 160]                        │
│   P3_context  [B, 256, 80, 80] → Upsample → [B, 256, 160, 160]
│     ↓                                                   │
│   Concat → [B, 512, 160, 160]                           │
│     ↓                                                   │
│   Conv(3×3) × 2 + BN + ReLU                             │
│     ↓                                                   │
│   Final Refinement Conv(3×3) + BN + SiLU               │
│     ↓                                                   │
│   P2_final [B, 256, 160, 160]                           │
└─────────────────────────────────────────────────────────┘
```

### Stage 2 输出

```python
enhanced_feats = [
    feat_p2_enhanced,  # [B, 256, 160, 160]  ← 小目标特征增强
    feat_p3,           # [B, 256, 80, 80]
    feat_p4,           # [B, 256, 40, 40]
    feat_p5,           # [B, 256, 20, 20]
]
```

---

## Stage 3: 解码器 (DEIMTransformer)

### 3.1 输入投影与展平

```
对每个特征层:

P2: [B, 256, 160, 160] → Flatten → [B, 25600, 256]
P3: [B, 256, 80, 80]   → Flatten → [B, 6400, 256]
P4: [B, 256, 40, 40]   → Flatten → [B, 1600, 256]
P5: [B, 256, 20, 20]   → Flatten → [B, 400, 256]

Concatenate所有层级:
    memory = [B, 25600+6400+1600+400, 256] = [B, 34000, 256]
    
spatial_shapes = [
    [160, 160],  # P2
    [80, 80],    # P3
    [40, 40],    # P4
    [20, 20],    # P5
]
```

### 3.2 生成Anchor

```
对每个特征层级 (P2, P3, P4, P5):
    
    grid_y, grid_x = meshgrid(H, W)
    grid_xy = (grid_xy + 0.5) / [W, H]  # 归一化到[0,1]
    wh = 0.05 × (2.0 ** level)  # P2: 0.05, P3: 0.1, P4: 0.2, P5: 0.4
    anchor = [grid_xy, wh, wh]  # [B, H×W, 4]
    
所有层级anchors拼接:
    anchors = [B, 34000, 4] (cx, cy, w, h in logit space)
    valid_mask = [B, 34000, 1]  # 过滤越界anchor
```

### 3.3 编码器输出头 (Encoder Output Heads)

```
memory [B, 34000, 256]
    ↓
┌────────────────────────────────────────┐
│ enc_score_head (Linear 256→80)         │
│ 预测每个anchor的类别分数                │
└────────────────────────────────────────┘
    ↓ enc_outputs_logits [B, 34000, 80]
    ↓
┌────────────────────────────────────────┐
│ enc_bbox_head (MLP 256→256→4)          │
│ 预测每个anchor的边界框偏移              │
└────────────────────────────────────────┘
    ↓ enc_bbox_delta [B, 34000, 4]
    ↓
enc_bbox_unact = enc_bbox_delta + anchors  # [B, 34000, 4]
```

### 3.4 小目标感知查询选择 (方案一)

```
输入:
    - memory [B, 34000, 256]
    - enc_outputs_logits [B, 34000, 80]
    - anchors [B, 34000, 4]
    - topk = 300

┌─────────────────────────────────────────────────────────┐
│ SmallObjectAwareQuerySelection                          │
│                                                         │
│ 1. 计算anchor面积:                                       │
│    anchors_sigmoid = sigmoid(anchors)                   │
│    area = w × h  → [B, 34000]                           │
│    small_object_mask = (area < 0.1)  # 小目标阈值       │
│                                                         │
│ 2. 小目标特征增强:                                       │
│    enhanced_memory = memory + small_object_proj(memory) │
│                      × small_object_mask.unsqueeze(-1)  │
│                                                         │
│ 3. 尺度感知权重:                                         │
│    scale_weights = sigmoid(scale_weight_pred(enhanced_memory))
│                    → [B, 34000]                         │
│                                                         │
│ 4. 小目标感知分数:                                       │
│    cls_scores = enc_outputs_logits.max(-1).values       │
│    object_aware_scores = cls_scores ×                   │
│                          (1.0 + scale_weights × small_object_mask)
│                                                         │
│ 5. 选择Top-K:                                            │
│    _, topk_ind = topk(object_aware_scores, k=300)       │
│                                                         │
│ 输出:                                                   │
│    - topk_memory [B, 300, 256]                          │
│    - topk_logits [B, 300, 80]                           │
│    - topk_anchors [B, 300, 4]                           │
└─────────────────────────────────────────────────────────┘
```

### 3.5 解码器输入准备

```
内容嵌入 (Content Embedding):
    content = topk_memory.detach()  # [B, 300, 256]

参考点 (Reference Points):
    ref_points_unact = enc_bbox_head(topk_memory) + topk_anchors
    # [B, 300, 4] in logit space
    ref_points_detach = sigmoid(ref_points_unact)  # [B, 300, 4]

位置编码:
    query_pos_embed = query_pos_head(ref_points_detach)
    # MLP: 4 → 256 → 256 → 256
    # [B, 300, 256]
```

### 3.6 Transformer解码器层

```
共6层解码器，每层结构:

输入:
    - target [B, 300, 256] (内容嵌入)
    - ref_points [B, 300, 4] (参考点)
    - value [B, 34000, 256] (编码器特征)
    - spatial_shapes (各层级尺寸)
    - query_pos_embed [B, 300, 256]

┌─────────────────────────────────────────────────────────┐
│ TransformerDecoderLayer (第i层)                         │
│                                                         │
│ 1. 自注意力 (Self-Attention):                            │
│    q = k = target + query_pos_embed                     │
│    target = target + SelfAttn(q, k, target)             │
│    target = RMSNorm(target)                             │
│                                                         │
│ 2. 交叉注意力 (Cross-Attention):                         │
│    ├─ 多尺度可变形注意力 (MSDeformableAttention)         │
│    │   采样点数: P2=8, P3/P4/P5=4 (方案二)               │
│    │   ├─ sampling_offsets: Linear(256 → total_points×2)│
│    │   ├─ attention_weights: Linear(256 → total_points) │
│    │   └─ 在value上采样并加权聚合                        │
│    ↓                                                    │
│    target = target + CrossAttn(target + query_pos_embed,│
│                                ref_points, value, ...)  │
│    target = RMSNorm(target) 或 Gateway                   │
│                                                         │
│ 3. 前馈网络 (FFN):                                       │
│    target = target + SwiGLUFFN(target)                  │
│    target = RMSNorm(target)                             │
│                                                         │
│ 输出: target [B, 300, 256]                              │
└─────────────────────────────────────────────────────────┘
```

### 3.7 检测头输出 (每解码器层)

```
第0层 (初始预测):
    pre_bboxes = sigmoid(pre_bbox_head(target) + inverse_sigmoid(ref_points_detach))
    # [B, 300, 4]
    pre_scores = score_head[0](target)  # [B, 300, 80]
    ref_points_initial = pre_bboxes.detach()

第i层 (迭代优化):
    # 边界框回归
    pred_corners = bbox_head[i](target + target_detach) + pred_corners_undetach
    # [B, 300, 4×(reg_max+1)] = [B, 300, 132]
    
    # 分布积分得到边界框
    inter_ref_bbox = distance2bbox(ref_points_initial, 
                                   integral(pred_corners, project), 
                                   reg_scale)
    # reg_scale: P2=2.0, P3=4.0, P4=6.0, P5=8.0 (方案三)
    
    # 分类分数
    scores = score_head[i](target)  # [B, 300, 80]
    
    # 增强LQE (方案五)
    if use_enhanced_lqe:
        scores = enhanced_lqe_layers[i](scores, pred_corners, target)
        # 内部: IoU预测 + 质量估计 + 原始LQE
    else:
        scores = lqe_layers[i](scores, pred_corners)
```

### 3.8 小目标专用检测分支 (方案四)

```
当 use_small_object_head=True 时:

decoder_features [B, 300, 256]
    ↓
┌─────────────────────────────────────────────────────────┐
│ SmallObjectDetectionBranch                              │
│                                                         │
│ 1. 特征增强:                                             │
│    enhanced = features + FeatureEnhance(features)       │
│    # FeatureEnhance: Linear(256→512) → ReLU → Dropout   │
│    #                   → Linear(512→256)                │
│                                                         │
│ 2. 多任务预测:                                           │
│    ├─ cls_head: Linear(256→80) → cls_logits            │
│    ├─ reg_head: MLP(256→256→132) → reg_corners         │
│    └─ objectness_head: Linear(256→1) → objectness      │
│                                                         │
│ 输出可以与主检测头融合或并行使用                         │
└─────────────────────────────────────────────────────────┘
```

### 3.9 输出生成

```
训练模式:
    输出包含:
    {
        'pred_logits': [B, 300, 80],      # 最终层分类分数
        'pred_boxes': [B, 300, 4],        # 最终层边界框 (cx, cy, w, h)
        'pred_corners': [B, 300, 132],    # 最终层分布参数
        'aux_outputs': [                   # 辅助输出 (前5层)
            {'pred_logits': ..., 'pred_boxes': ..., ...},
            ...
        ],
        'enc_aux_outputs': [...],          # 编码器辅助输出
        'pre_outputs': {...},              # 初始预测
    }

推理模式:
    输出仅包含:
    {
        'pred_logits': [B, 300, 80],
        'pred_boxes': [B, 300, 4],
    }
```

---

## 完整数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                         输入图像 [B, 3, 640, 640]                    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: 骨干网络 (Backbone)                                        │
│ ├─ Spatial Prior Module: 生成多尺度特征 (P2-P5)                      │
│ ├─ DINOv3 Transformer: 提取语义特征                                  │
│ └─ 特征融合投影: 输出 [P2, P3, P4, P5]                              │
│     ├─ P2: [B, 256, 160, 160]  (stride=4)                          │
│     ├─ P3: [B, 256, 80, 80]    (stride=8)                          │
│     ├─ P4: [B, 256, 40, 40]    (stride=16)                         │
│     └─ P5: [B, 256, 20, 20]    (stride=32)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2: 编码器 (HybridEncoder)                                     │
│ ├─ 输入投影: 保持通道数256                                           │
│ ├─ 内部处理: CSPLayer增强特征                                        │
│ ├─ 跨尺度融合: 自上而下 + 自下而上                                   │
│ └─ 小目标增强: SmallObjectEnhancementModule                          │
│     ├─ P2: CBAM + DepthwiseConv (小目标特征增强)                     │
│     ├─ P3: ContextAggregation (多尺度上下文)                         │
│     └─ P2+P3融合: 增强小目标表示                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3: 解码器 (DEIMTransformer)                                   │
│ ├─ 输入投影: 展平为 memory [B, 34000, 256]                          │
│ ├─ Anchor生成: [B, 34000, 4]                                        │
│ ├─ 编码器预测: 类别分数 + 边界框偏移                                 │
│ ├─ 小目标感知查询选择: 选择300个查询 (优先小目标)                    │
│ ├─ Transformer解码器 (6层):                                         │
│ │   ├─ 自注意力: 查询间交互                                          │
│ │   ├─ 交叉注意力: 多尺度可变形注意力 (P2:8点, 其他:4点)             │
│ │   ├─ 尺度自适应回归: 不同层使用不同reg_scale                       │
│ │   └─ 增强LQE: IoU预测 + 质量估计                                   │
│ ├─ 小目标检测分支: 专用分类/回归头                                   │
│ └─ 输出: pred_logits + pred_boxes                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      检测结果                                        │
│              {'pred_logits': [B, 300, 80],                          │
│               'pred_boxes': [B, 300, 4]}                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 关键维度速查表

| 阶段 | 张量 | 维度 | 说明 |
|------|------|------|------|
| 输入 | image | [B, 3, 640, 640] | 输入图像 |
| Backbone | P2 | [B, 256, 160, 160] | stride=4 |
| | P3 | [B, 256, 80, 80] | stride=8 |
| | P4 | [B, 256, 40, 40] | stride=16 |
| | P5 | [B, 256, 20, 20] | stride=32 |
| Encoder | memory | [B, 34000, 256] | 展平后特征 |
| | anchors | [B, 34000, 4] | 所有anchor |
| Query Selection | topk_memory | [B, 300, 256] | 选中的300个查询 |
| | topk_anchors | [B, 300, 4] | 选中的anchor |
| Decoder | target | [B, 300, 256] | 每层输出 |
| | pred_corners | [B, 300, 132] | 4×(32+1)分布 |
| Output | pred_logits | [B, 300, 80] | 分类分数 |
| | pred_boxes | [B, 300, 4] | 边界框坐标 |

---

## 小目标检测改进点总结

| 方案 | 位置 | 改进内容 | 影响 |
|------|------|----------|------|
| 方案一 | 查询选择 | 小目标感知分数加权 | 提升小目标被选中概率 |
| 方案二 | 交叉注意力 | P2层8个采样点 | 更精细的小目标特征采样 |
| 方案三 | 回归头 | 尺度自适应reg_scale | P2:2.0, P3:4.0, P4:6.0, P5:8.0 |
| 方案四 | 检测分支 | 小目标专用分支 | 增强小目标分类和回归 |
| 方案五 | LQE | IoU感知分类 | 分类分数反映定位质量 |
