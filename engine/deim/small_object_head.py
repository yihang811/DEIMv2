"""
DEIMv2 Small Object Detection Head Enhancement
Enhanced detection head modules for small object detection in UAV imagery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .deim_utils import MLP
from .dfine_decoder import LQE


class SmallObjectAwareQuerySelection(nn.Module):
    """
    小目标感知查询选择模块
    根据anchor大小调整选择权重，优先选择小目标区域
    """
    def __init__(self, hidden_dim, num_classes, small_object_threshold=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.small_object_threshold = small_object_threshold  # 小目标尺寸阈值
        
        # 小目标特征增强层
        self.small_object_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 尺度感知权重预测
        self.scale_weight_pred = nn.Linear(hidden_dim, 1)
        
    def forward(self, memory, outputs_logits, anchors, topk, training=True):
        """
        Args:
            memory: [bs, num_anchors, hidden_dim]
            outputs_logits: [bs, num_anchors, num_classes]
            anchors: [bs, num_anchors, 4] (cx, cy, w, h) in logit space
            topk: number of queries to select
        Returns:
            topk_memory, topk_logits, topk_anchors
        """
        # 计算anchor的面积（在sigmoid空间中）
        anchors_sigmoid = torch.sigmoid(anchors)  # [bs, num_anchors, 4]
        anchor_wh = anchors_sigmoid[..., 2:]  # [bs, num_anchors, 2]
        anchor_areas = anchor_wh[..., 0] * anchor_wh[..., 1]  # [bs, num_anchors]
        
        # 小目标掩码
        small_object_mask = (anchor_areas < self.small_object_threshold).float()
        
        # 原始分类分数
        cls_scores = outputs_logits.max(-1).values  # [bs, num_anchors]
        
        # 小目标特征增强
        enhanced_memory = memory + self.small_object_proj(memory) * small_object_mask.unsqueeze(-1)
        
        # 尺度感知权重 [bs, num_anchors, 1]
        scale_weights = torch.sigmoid(self.scale_weight_pred(enhanced_memory)).squeeze(-1)
        
        # 小目标感知分数 = 分类分数 * (1 + 小目标权重 * 小目标掩码)
        # 提升小目标的分数，使其更容易被选中
        object_aware_scores = cls_scores * (1.0 + scale_weights * small_object_mask)
        
        # 选择topk
        _, topk_ind = torch.topk(object_aware_scores, topk, dim=-1)
        
        #  gather selected features
        topk_anchors = anchors.gather(
            dim=1, 
            index=topk_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1])
        )
        topk_logits = outputs_logits.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
        ) if training else None
        topk_memory = memory.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )
        
        return topk_memory, topk_logits, topk_anchors


class ScaleAdaptiveRegHead(nn.Module):
    """
    尺度自适应回归头
    为不同尺度特征层使用不同的回归范围
    """
    def __init__(self, hidden_dim, reg_max, num_levels=4, 
                 reg_scales=[2.0, 4.0, 6.0, 8.0], act='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reg_max = reg_max
        self.num_levels = num_levels
        
        # 为每个尺度创建独立的回归头
        self.bbox_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, act=act)
            for _ in range(num_levels)
        ])
        
        # 可学习的尺度回归参数
        self.reg_scales = nn.ParameterList([
            nn.Parameter(torch.tensor([scale]), requires_grad=True)
            for scale in reg_scales
        ])
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for bbox_head in self.bbox_heads:
            init.constant_(bbox_head.layers[-1].weight, 0)
            init.constant_(bbox_head.layers[-1].bias, 0)
    
    def forward(self, features, level_idx):
        """
        Args:
            features: [bs, num_queries, hidden_dim]
            level_idx: which feature level to use
        Returns:
            pred_corners: [bs, num_queries, 4 * (reg_max + 1)]
            reg_scale: scalar
        """
        pred_corners = self.bbox_heads[level_idx](features)
        reg_scale = self.reg_scales[level_idx]
        return pred_corners, reg_scale


class SmallObjectDetectionBranch(nn.Module):
    """
    小目标专用检测分支
    专门处理小尺寸目标的检测
    """
    def __init__(self, hidden_dim, num_classes, reg_max, act='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 小目标特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 小目标分类头（更精细的分类）
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 小目标回归头（更精细的回归）
        self.reg_head = MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, act=act)
        
        # 小目标置信度预测
        self.objectness_head = nn.Linear(hidden_dim, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        bias = -torch.log(torch.tensor((1 - 0.01) / 0.01))
        init.constant_(self.cls_head[-1].bias, bias)
        init.constant_(self.reg_head.layers[-1].weight, 0)
        init.constant_(self.reg_head.layers[-1].bias, 0)
        init.constant_(self.objectness_head.bias, bias)
        
    def forward(self, features):
        """
        Args:
            features: [bs, num_queries, hidden_dim]
        Returns:
            cls_logits: [bs, num_queries, num_classes]
            reg_corners: [bs, num_queries, 4 * (reg_max + 1)]
            objectness: [bs, num_queries, 1]
        """
        # 特征增强
        enhanced_features = features + self.feature_enhance(features)
        
        # 分类、回归、置信度预测
        cls_logits = self.cls_head(enhanced_features)
        reg_corners = self.reg_head(enhanced_features)
        objectness = torch.sigmoid(self.objectness_head(enhanced_features))
        
        return cls_logits, reg_corners, objectness


class IoUAwareClassification(nn.Module):
    """
    IoU感知分类优化模块
    预测IoU并用于加权分类分数
    """
    def __init__(self, hidden_dim, num_classes, k=4, reg_max=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.k = k
        self.reg_max = reg_max
        
        # IoU预测头 - 输入维度: hidden_dim + 4 * (k + 1)
        # stat的维度是 4 * (k + 1)，因为4个边，每个边有k+1个统计值
        stat_dim = 4 * (k + 1)
        self.iou_head = nn.Sequential(
            nn.Linear(hidden_dim + stat_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 质量估计（基于分布的集中度）
        self.quality_head = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        init.constant_(self.iou_head[-1].bias, 0)
        init.constant_(self.iou_head[-1].weight, 0)
        init.constant_(self.quality_head[-1].bias, 0)
        
    def forward(self, cls_scores, pred_corners, decoder_features):
        """
        Args:
            cls_scores: [bs, num_queries, num_classes]
            pred_corners: [bs, num_queries, 4 * (reg_max + 1)]
            decoder_features: [bs, num_queries, hidden_dim]
        Returns:
            refined_scores: [bs, num_queries, num_classes]
        """
        B, L, _ = pred_corners.size()
        
        # 计算分布统计信息
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        stat = stat.reshape(B, L, -1)
        
        # 质量分数（分布越集中，质量越高）
        quality = torch.sigmoid(self.quality_head(stat))
        
        # IoU预测
        iou_input = torch.cat([decoder_features, stat], dim=-1)
        pred_iou = torch.sigmoid(self.iou_head(iou_input))
        
        # 融合：分类分数 * IoU * 质量
        refined_scores = cls_scores * pred_iou * quality
        
        return refined_scores


class EnhancedLQE(nn.Module):
    """
    增强的位置质量估计模块
    结合IoU预测和分布质量
    """
    def __init__(self, k, hidden_dim, num_layers, reg_max, num_classes, act='relu'):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.num_classes = num_classes
        
        # 原始LQE
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        
        # IoU感知分类
        self.iou_aware_cls = IoUAwareClassification(hidden_dim, num_classes, k, reg_max)
        
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)
        
    def forward(self, scores, pred_corners, decoder_features=None):
        """
        Args:
            scores: [bs, num_queries, num_classes] or [bs, num_queries, 1]
            pred_corners: [bs, num_queries, 4 * (reg_max + 1)]
            decoder_features: [bs, num_queries, hidden_dim] (optional)
        Returns:
            refined_scores: quality-enhanced scores
        """
        B, L, _ = pred_corners.size()
        
        # 计算分布统计
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        
        # 位置质量分数
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        
        # 如果提供了decoder特征，使用IoU感知分类
        if decoder_features is not None and scores.size(-1) == self.num_classes:
            scores = self.iou_aware_cls(scores, pred_corners, decoder_features)
        
        return scores + quality_score


class SmallObjectDecoderLayer(nn.Module):
    """
    小目标专用解码器层
    增强对小目标特征的提取能力
    """
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024, 
                 dropout=0., n_levels=4, n_points=4):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 小目标感知交叉注意力（使用更多采样点）
        from .dfine_decoder import MSDeformableAttention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, [n_points * 2] + [n_points] * (n_levels - 1)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 小目标特征门控
        self.small_object_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, target, reference_points, value, spatial_shapes, 
                attn_mask=None, query_pos_embed=None, small_object_mask=None):
        """
        Args:
            small_object_mask: [bs, num_queries, 1] 标识哪些query是小目标
        """
        # 自注意力
        q = k = target if query_pos_embed is None else target + query_pos_embed
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        
        # 交叉注意力
        target2 = self.cross_attn(
            target + query_pos_embed if query_pos_embed is not None else target,
            reference_points,
            value,
            spatial_shapes
        )
        
        # 小目标特征增强
        if small_object_mask is not None:
            gate = self.small_object_gate(torch.cat([target, target2], dim=-1))
            target2 = target2 * (1 + small_object_mask * gate)
        
        target = target + self.dropout2(target2)
        target = self.norm2(target)
        
        # FFN
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        target = target + self.dropout4(target2)
        target = self.norm3(target)
        
        return target
