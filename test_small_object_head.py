"""
测试小目标检测头增强模块
验证所有五个方案是否正确集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List
import sys
import os

# 添加项目路径
sys.path.insert(0, 'D:/desktop/在线学习/DEIMv2-main/DEIMv2')
os.chdir('D:/desktop/在线学习/DEIMv2-main/DEIMv2')

# 手动定义必要的辅助函数和类
def bias_init_with_prob(prior_prob):
    """Initialize bias with probability"""
    bias = -torch.log((1 - prior_prob) / prior_prob)
    return bias

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = F.relu if act == 'relu' else F.silu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# 读取并执行small_object_head.py的内容（替换相对导入）
small_object_head_path = 'D:/desktop/在线学习/DEIMv2-main/DEIMv2/engine/deim/small_object_head.py'
with open(small_object_head_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换相对导入
content = content.replace('from .deim_utils import MLP', '# from .deim_utils import MLP')
content = content.replace('from .dfine_decoder import LQE', '# from .dfine_decoder import LQE')
content = content.replace('from .dfine_utils import weighting_function, distance2bbox', '# from .dfine_utils import weighting_function, distance2bbox')
content = content.replace('from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP', '# from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP')
content = content.replace('from .dfine_decoder import MSDeformableAttention, LQE, Integral', '# from .dfine_decoder import MSDeformableAttention, LQE, Integral')

# 执行修改后的代码
exec(content, globals())

def test_small_object_aware_query_selection():
    """测试方案一：小目标感知查询选择"""
    print("=" * 60)
    print("测试方案一：小目标感知查询选择")
    print("=" * 60)
    
    batch_size = 2
    num_anchors = 100
    hidden_dim = 256
    num_classes = 80
    topk = 10
    
    module = SmallObjectAwareQuerySelection(hidden_dim, num_classes, small_object_threshold=0.1)
    
    memory = torch.randn(batch_size, num_anchors, hidden_dim)
    outputs_logits = torch.randn(batch_size, num_anchors, num_classes)
    anchors = torch.randn(batch_size, num_anchors, 4)  # logit space
    
    topk_memory, topk_logits, topk_anchors = module(memory, outputs_logits, anchors, topk, training=True)
    
    print(f"  输入: memory {memory.shape}, logits {outputs_logits.shape}, anchors {anchors.shape}")
    print(f"  输出: topk_memory {topk_memory.shape}, topk_logits {topk_logits.shape}, topk_anchors {topk_anchors.shape}")
    print(f"  [OK] 小目标感知查询选择模块测试通过\n")

def test_scale_adaptive_reg_head():
    """测试方案三：尺度自适应回归头"""
    print("=" * 60)
    print("测试方案三：尺度自适应回归头")
    print("=" * 60)
    
    batch_size = 2
    num_queries = 10
    hidden_dim = 256
    reg_max = 32
    num_levels = 4
    
    module = ScaleAdaptiveRegHead(hidden_dim, reg_max, num_levels, 
                                   reg_scales=[2.0, 4.0, 6.0, 8.0])
    
    features = torch.randn(batch_size, num_queries, hidden_dim)
    
    for level_idx in range(num_levels):
        pred_corners, reg_scale = module(features, level_idx)
        print(f"  Level {level_idx}: pred_corners {pred_corners.shape}, reg_scale {reg_scale.item():.2f}")
    
    print(f"  [OK] 尺度自适应回归头测试通过\n")

def test_small_object_detection_branch():
    """测试方案四：小目标专用检测分支"""
    print("=" * 60)
    print("测试方案四：小目标专用检测分支")
    print("=" * 60)
    
    batch_size = 2
    num_queries = 10
    hidden_dim = 256
    num_classes = 80
    reg_max = 32
    
    module = SmallObjectDetectionBranch(hidden_dim, num_classes, reg_max)
    
    features = torch.randn(batch_size, num_queries, hidden_dim)
    
    cls_logits, reg_corners, objectness = module(features)
    
    print(f"  输入: features {features.shape}")
    print(f"  输出: cls_logits {cls_logits.shape}, reg_corners {reg_corners.shape}, objectness {objectness.shape}")
    print(f"  [OK] 小目标专用检测分支测试通过\n")

def test_iou_aware_classification():
    """测试方案五：IoU感知分类"""
    print("=" * 60)
    print("测试方案五：IoU感知分类")
    print("=" * 60)
    
    batch_size = 2
    num_queries = 10
    hidden_dim = 256
    num_classes = 80
    reg_max = 32
    
    module = IoUAwareClassification(hidden_dim, num_classes, k=4, reg_max=reg_max)
    
    cls_scores = torch.randn(batch_size, num_queries, num_classes)
    pred_corners = torch.randn(batch_size, num_queries, 4 * (reg_max + 1))
    decoder_features = torch.randn(batch_size, num_queries, hidden_dim)
    
    refined_scores = module(cls_scores, pred_corners, decoder_features)
    
    print(f"  输入: cls_scores {cls_scores.shape}, pred_corners {pred_corners.shape}")
    print(f"  输出: refined_scores {refined_scores.shape}")
    print(f"  [OK] IoU感知分类模块测试通过\n")

def test_enhanced_lqe():
    """测试增强LQE模块"""
    print("=" * 60)
    print("测试增强LQE模块")
    print("=" * 60)
    
    batch_size = 2
    num_queries = 10
    hidden_dim = 256
    num_classes = 80
    reg_max = 32
    
    module = EnhancedLQE(k=4, hidden_dim=hidden_dim, num_layers=2, reg_max=reg_max, 
                         num_classes=num_classes, act='relu')
    
    scores = torch.randn(batch_size, num_queries, num_classes)
    pred_corners = torch.randn(batch_size, num_queries, 4 * (reg_max + 1))
    decoder_features = torch.randn(batch_size, num_queries, hidden_dim)
    
    refined_scores = module(scores, pred_corners, decoder_features)
    
    print(f"  输入: scores {scores.shape}, pred_corners {pred_corners.shape}")
    print(f"  输出: refined_scores {refined_scores.shape}")
    print(f"  [OK] 增强LQE模块测试通过\n")

def test_deim_transformer_with_small_object_head():
    """测试集成后的DEIMTransformer"""
    print("=" * 60)
    print("测试集成后的DEIMTransformer")
    print("=" * 60)
    
    from engine.deim.deim_decoder import DEIMTransformer
    
    # 创建配置
    config = {
        'num_classes': 80,
        'hidden_dim': 256,
        'num_queries': 300,
        'feat_channels': [256, 256, 256, 256],
        'feat_strides': [4, 8, 16, 32],
        'num_levels': 4,
        'num_points': [8, 4, 4, 4],  # 方案二：多尺度采样点
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'eval_spatial_size': [640, 640],
        'reg_max': 32,
        # 小目标检测增强配置
        'use_small_object_head': True,      # 方案一、四
        'small_object_threshold': 0.1,
        'use_scale_adaptive_reg': True,      # 方案三
        'reg_scales_per_level': [2.0, 4.0, 6.0, 8.0],
        'use_enhanced_lqe': True,            # 方案五
        'num_points_per_scale': [8, 4, 4, 4],
    }
    
    try:
        model = DEIMTransformer(**config)
        print(f"  [OK] DEIMTransformer创建成功")
        print(f"  - 使用小目标感知查询选择: {model.use_small_object_head}")
        print(f"  - 使用尺度自适应回归: {model.use_scale_adaptive_reg}")
        print(f"  - 使用增强LQE: {model.use_enhanced_lqe}")
        print(f"  - 多尺度采样点: {model.num_points_per_scale}")
        
        # 测试前向传播
        batch_size = 1
        feats = [
            torch.randn(batch_size, 256, 160, 160),  # P2: stride 4
            torch.randn(batch_size, 256, 80, 80),    # P3: stride 8
            torch.randn(batch_size, 256, 40, 40),    # P4: stride 16
            torch.randn(batch_size, 256, 20, 20),    # P5: stride 32
        ]
        
        model.eval()
        with torch.no_grad():
            output = model(feats)
        
        print(f"  [OK] 前向传播测试通过")
        print(f"  - 输出pred_logits: {output['pred_logits'].shape}")
        print(f"  - 输出pred_boxes: {output['pred_boxes'].shape}")
        
    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DEIMv2 小目标检测头增强模块测试")
    print("=" * 60 + "\n")
    
    # 测试各个模块
    test_small_object_aware_query_selection()
    test_scale_adaptive_reg_head()
    test_small_object_detection_branch()
    test_iou_aware_classification()
    test_enhanced_lqe()
    test_deim_transformer_with_small_object_head()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
