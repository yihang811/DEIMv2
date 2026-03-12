"""
DEIMv2 Small Object Detection Enhancement Modules
Enhanced modules for detecting small objects in UAV imagery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SmallObjectFeatureEnhancer(nn.Module):
    """
    Small Object Feature Enhancer
    Uses CBAM attention and depthwise separable convolution to enhance P2 features
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction)
        self.dw_conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1)
        self.dw_conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Attention enhancement
        x_attn = self.cbam(x)
        # Feature extraction
        x_conv = self.dw_conv1(x_attn)
        x_conv = self.dw_conv2(x_conv)
        # Residual connection
        return x_conv + self.shortcut(x)


class ContextAggregation(nn.Module):
    """
    Context Aggregation Module
    Uses multi-scale dilated convolutions to capture context information
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 3, 6]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        for dilation in dilations:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(dilations), 
                         kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(in_channels // len(dilations)),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        return self.fusion(concat)


class SmallObjectFusion(nn.Module):
    """
    Small Object Fusion Module
    Fuses P2 and P3 features for better small object detection
    """
    def __init__(self, p2_channels, p3_channels, out_channels):
        super().__init__()
        # P2 feature processing
        self.p2_conv = nn.Sequential(
            nn.Conv2d(p2_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # P3 feature processing (upsample to P2 resolution)
        self.p3_conv = nn.Sequential(
            nn.Conv2d(p3_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, p2_feat, p3_feat):
        # Process P2 features
        p2_out = self.p2_conv(p2_feat)
        
        # Process and upsample P3 features
        p3_out = self.p3_conv(p3_feat)
        p3_up = F.interpolate(p3_out, size=p2_out.shape[2:], mode='nearest')
        
        # Concatenate and fuse
        concat = torch.cat([p2_out, p3_up], dim=1)
        return self.fusion_conv(concat)


class SmallObjectEnhancementModule(nn.Module):
    """
    Complete Small Object Enhancement Module
    Integrates feature enhancement, context aggregation, and fusion
    """
    def __init__(self, p2_channels, p3_channels, out_channels=256):
        super().__init__()
        # P2 feature enhancement
        self.p2_enhancer = SmallObjectFeatureEnhancer(p2_channels, out_channels)
        
        # Context aggregation for P3
        self.context_agg = ContextAggregation(p3_channels, out_channels)
        
        # P2-P3 fusion
        self.fusion = SmallObjectFusion(out_channels, out_channels, out_channels)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, p2_feat, p3_feat):
        # Enhance P2 features
        p2_enhanced = self.p2_enhancer(p2_feat)
        
        # Aggregate context from P3
        p3_context = self.context_agg(p3_feat)
        
        # Fuse enhanced P2 with P3 context
        fused = self.fusion(p2_enhanced, p3_context)
        
        # Final refinement
        return self.refine(fused)
