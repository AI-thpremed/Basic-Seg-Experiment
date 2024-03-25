# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN
#import contextlib
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from collections import OrderedDict
import re
import math
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        """设置卷积核大小为5"""
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)  #mednext换成groupnorm
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 include_head=True  # Add this line                 
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers stem对初始图像进行处理，以及三个下采样层，至于为什么要采用4*4的卷积来直接吧输入图像缩小到四倍则不清楚，应该是为了适用于自监督算法，所以这里如果缩小到两倍是不是更好
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        if include_head:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = None 

        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


class FF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FF, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)

    def forward(self, fi_minus1, fi, fi_plus1):
        f_prime_plus1 = self.upsample(self.conv1x1(fi_plus1))
        f_prime_minus1 = self.conv1x1(self.dwconv(fi_minus1))
        
        f_double_prime_plus1 = f_prime_plus1 * fi
        f_double_prime_minus1 = f_prime_minus1 * fi
        
        f_pff = torch.cat((f_double_prime_minus1, f_double_prime_plus1), dim=1)
        return f_pff

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x).view(x.size(0), -1)
        max_pool_out = self.max_pool(x).view(x.size(0), -1)
        
        avg_pool_out1 = self.fc1(avg_pool_out)
        avg_pool_out = self.fc2(self.relu(avg_pool_out1))
        max_pool_out = self.fc2(self.relu(self.fc1(max_pool_out)))

        channel_attention = self.sigmoid(avg_pool_out + max_pool_out)
        channel_attention = channel_attention.view(x.size(0), x.size(1), 1, 1)
        
        return (x * channel_attention)

class ChannelAttention1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x).view(x.size(0), -1)
        max_pool_out = self.max_pool(x).view(x.size(0), -1)
        
        avg_pool_out1 = self.fc1(avg_pool_out)
        avg_pool_out = self.fc2(self.relu(avg_pool_out1))
        max_pool_out = self.fc2(self.relu(self.fc1(max_pool_out)))

        channel_attention = self.sigmoid(avg_pool_out + max_pool_out)
        channel_attention = channel_attention.view(x.size(0), x.size(1), 1, 1)
        
        return (x * channel_attention),channel_attention 
class ModifiedChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ModifiedChannelAttention, self).__init__()
        self.channel_attention = ChannelAttention1(in_channels)

    def split_into_parts(self, x, parts=4):
        part_size = x.size(2) // parts
        return [x[:, :, i*part_size: (i+1)*part_size, :] for i in range(parts)]

    def forward(self, x):
        # Split the input into 4 parts
        parts = self.split_into_parts(x)

        # Apply channel attention to each part
        attention_weights = [self.channel_attention(part)[1] for part in parts]
        
        # Concatenate along the height dimension
        concatenated_weights = torch.cat(attention_weights, dim=2)
        
        # Apply softmax for normalization along the height dimension
        concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), concatenated_weights.size(1), -1), dim=2).view_as(concatenated_weights)
        
        # Resplit the concatenated_weights to apply them to the corresponding parts
        split_weights = self.split_into_parts(concatenated_weights)

        # Multiply the weights with the corresponding parts
        weighted_parts = [part * weight for part, weight in zip(parts, split_weights)]
        
        # Concatenate the weighted parts back into a complete image
        output = torch.cat(weighted_parts, dim=2)
        
        return output
# class ModifiedChannelAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(ModifiedChannelAttention, self).__init__()
#         self.channel_attention = ChannelAttention1(in_channels)

#     def split_into_parts(self, x, parts=4):
#         part_size = x.size(2) // parts
#         return [x[:, :, i*part_size: (i+1)*part_size, :] for i in range(parts)]

#     def forward(self, x):
#         parts = self.split_into_parts(x)

#         attention_weights = [self.channel_attention(part)[1] for part in parts]

#         # Concatenate along the height dimension
#         concatenated_weights = torch.cat(attention_weights, dim=2)

#         # Apply softmax for normalization along the height and width dimensions
#         concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), concatenated_weights.size(1), -1), dim=2).view_as(concatenated_weights)

#         return x * concatenated_weights
# 使用示例
# input_tensor = torch.randn(32, 64, 32, 32)  # 假设输入张量具有形状 [batch_size, channels, height, width]
# channel_attention = ChannelAttention(64)
# output_tensor = channel_attention(input_tensor)

class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv(x1)
        spatial_attention = self.sigmoid(x1)
        
        return (x * spatial_attention)###,spatial_attention
"""赋值权重方法"""
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7, height=512, width=512):
#         super(SpatialAttention, self).__init__()
        
#         # 1. 创建初始化注意力权重矩阵
#         # self.prior_attention = torch.ones(1, 1, height, width) * 0.5
#         # h_start, h_end = int(1/5 * height), int(4/5 * height)
#         # self.prior_attention[0, 0, h_start:h_end, :] = 1
#         # 初始化卷积权重
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#         # out_channels, in_channels, kernel_height, kernel_width = self.conv.weight.data.shape
#         # prior_attention = torch.ones(1, 1, kernel_height, kernel_width) * 0.5 
#         # h_start, h_end = int(1/5 * kernel_height), int(4/5 * kernel_height)
#         # prior_attention[0, 0, h_start:h_end, :] = 1

#         # self.conv.weight.data.copy_(prior_attention)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, h, w = x.size()
#         prior_attention = torch.ones(1, 1, h, w).to(x.device) * 0.5
#         h_start, h_end = int(1/5 * h), int(4/5 * h)
#         prior_attention[0, 0, h_start:h_end, :] = 1

#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x1 = torch.cat([avg_out, max_out], dim=1)
#         x1 = self.conv(x1)
#         spatial_attention = self.sigmoid(x1)
#         spatial_attention = spatial_attention * prior_attention
#         return (x * spatial_attention)

"""四卷积空间注意力方法"""
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        # 定义四个不同大小的卷积核
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # 最终的卷积操作
        self.final_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False)
        # self.final_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False, dilation=2)  # Using dilation

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x1 = torch.cat([avg_out, max_out], dim=1)

        # 使用四个卷积核处理输入
        x1_conv1 = self.conv1(x1)
        x1_conv3 = self.conv3(x1)
        x1_conv5 = self.conv5(x1)
        x1_conv7 = self.conv7(x1)

        # 将四个输出沿通道维度拼接
        x_concat = torch.cat([x1_conv1, x1_conv3, x1_conv5, x1_conv7], dim=1)

        # 使用最终的卷积层处理拼接后的输出
        x_final = self.final_conv(x_concat)
        spatial_attention = self.sigmoid(x_final)

        return (x * spatial_attention),spatial_attention
    
"""纯四部分空间注意力方法"""
class ModifiedSpatialAttention(nn.Module):
    def __init__(self):
        super(ModifiedSpatialAttention, self).__init__()
        self.spatial_attention = SpatialAttention()

    def split_into_parts(self, x, parts=4):
        # Compute the size for each part
        part_size = x.size(2) // parts
        return [x[:, :, i*part_size: (i+1)*part_size, :] for i in range(parts)]
        
    def forward(self, x):
        parts = self.split_into_parts(x)

        # Apply spatial attention to each part
        attention_weights = [self.spatial_attention(part)[1] for part in parts]
        
        # Concatenate along the height dimension
        concatenated_weights = torch.cat(attention_weights, dim=2)
        
        # Apply softmax for normalization along the height and width dimensions
        concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), -1), dim=1).view_as(concatenated_weights)
        
        return x * concatenated_weights

"""最终的高斯权重方法"""
# class ModifiedSpatialAttention(nn.Module):
#     def __init__(self):
#         super(ModifiedSpatialAttention, self).__init__()
#         self.spatial_attention = SpatialAttention()

#     def create_gaussian_weights(self, h, w):
#         y = np.linspace(-h // 2, h // 2, h)
#         x = np.linspace(-w // 2, w // 2, w)
#         y, x = np.meshgrid(y, x, indexing='ij')
#         dist = np.sqrt(x * x + y * y)
#         std = min(h, w) * 0.25  # std 设为高/宽的四分之一
#         gaussian_weights = np.exp(-(dist ** 2) / (2 * std ** 2))
#         gaussian_weights = torch.from_numpy(gaussian_weights).float().unsqueeze(0).unsqueeze(0)
#         return gaussian_weights

#     def split_into_parts(self, x, parts=4):
#         # Compute the size for each part
#         part_size = x.size(2) // parts
#         return [x[:, :, i*part_size: (i+1)*part_size, :] for i in range(parts)]
        
#     def forward(self, x):
#         _, _, height, width = x.size()
#         gaussian_weights = self.create_gaussian_weights(height, width)
#         gaussian_weights = gaussian_weights.to(x.device)  # Ensure weights are on the same device as x

#         parts = self.split_into_parts(x)
#         # Apply spatial attention to each part
#         attention_weights = [self.spatial_attention(part)[1] for part in parts]
#         # Concatenate along the height dimension
#         concatenated_weights = torch.cat(attention_weights, dim=2)
#         # Apply softmax for normalization along the height and width dimensions
#         # concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), -1), dim=1).view_as(concatenated_weights)

#         # flattened_weights = concatenated_weights.view(concatenated_weights.size(0), -1)
#         # softmax_weights = F.softmax(flattened_weights, dim=1)
#         # concatenated_weights = softmax_weights.view_as(concatenated_weights)

#         # Multiply concatenated weights by the gaussian weights
#         concatenated_weights = concatenated_weights * gaussian_weights
#         concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), -1), dim=1).view_as(concatenated_weights)
#         #concatenated_weights *= gaussian_weights
#         return x * concatenated_weights

# class ModifiedSpatialAttention(nn.Module):
#     def __init__(self):
#         super(ModifiedSpatialAttention, self).__init__()
#         self.spatial_attention = SpatialAttention()
#         self.gaussian_weights = self.create_gaussian_weights(input_height, input_width)
#         self.gaussian_weights = nn.Parameter(self.gaussian_weights, requires_grad=False)

#     def create_gaussian_weights(self, h, w):
#         y = np.linspace(-h // 2, h // 2, h)
#         x = np.linspace(-w // 2, w // 2, w)
#         y, x = np.meshgrid(y, x, indexing='ij')
#         dist = np.sqrt(x * x + y * y)
#         std = min(h, w) * 0.25  # std 设为高/宽的四分之一
#         gaussian_weights = np.exp(-(dist ** 2) / (2 * std ** 2))
#         gaussian_weights = torch.from_numpy(gaussian_weights).float().unsqueeze(0).unsqueeze(0)
#         return gaussian_weights

#     def split_into_parts(self, x, parts=4):
#         # Compute the size for each part
#         part_size = x.size(2) // parts
#         return [x[:, :, i*part_size: (i+1)*part_size, :] for i in range(parts)]
        
#     def forward(self, x):
#         _, _, height, width = x.size()
#         parts = self.split_into_parts(x)
#         # Apply spatial attention to each part
#         attention_weights = [self.spatial_attention(part)[1] for part in parts]
#         # Concatenate along the height dimension
#         concatenated_weights = torch.cat(attention_weights, dim=2)
#         # Apply softmax for normalization along the height and width dimensions
#         concatenated_weights = F.softmax(concatenated_weights.view(concatenated_weights.size(0), -1), dim=1).view_as(concatenated_weights)
#         # Multiply concatenated weights by the gaussian weights
#         concatenated_weights *= self.gaussian_weights
#         return x * concatenated_weights



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)+x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.eca = eca_layer(out_features, 3)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        x = self.eca(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
        
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
       
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #q = torch.nn.functional.normalize(q, dim=-1)  
        #k = torch.nn.functional.normalize(k, dim=-1)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        '''
        B,h,n1,n2 = attn.shape
        W1 = torch.mean(attn, 1, True)
        W = torch.sigmoid(W1)
        attn = attn * W
        '''
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            #attn = attn + attn.transpose(-2, -1)

        attn = self.attn_drop(attn) 
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        ##### channel dimension
        '''
        if mask is  None:
            attn_c = (k.transpose(-2, -1) @ q)  
            attn_c = self.softmax(attn_c)
            x_c = (v @ attn_c).transpose(1, 2).reshape(B_, N, C)  # channel dimension
            x = x + x_c
        '''
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
            # print("dim:", dim),
            # print("mult:", mult),
            # nn.Conv2d(1536, 768, kernel_size=3, padding=1),

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(dim, dim * mult,1,1,bias=False),
            nn.GELU(),
            nn.Conv2d(dim*mult, dim*mult, 3, 1, 1, bias=False, groups=dim*mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim,1,1,bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        B,N,C = x.shape
        H= W = int(np.sqrt(N))
        x = x.view(B,H,W,C).permute(0,3,1,2 )
        out = self.net(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.FeedForward = FeedForward(dim=dim, mult=mlp_ratio)
        
        #### channel ##
        #self.C_Att = ChannelBlock(dim=dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop_path=drop_path,norm_layer=nn.LayerNorm)
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        #print(x.shape)        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
      
        """
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        """
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        """gaileshangmian zheyige bufen de neirong"""
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        """mlp换成feedforward"""
        x = x + self.drop_path(self.FeedForward(self.norm2(x)))
        # Channl
        #x = self.C_Att(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)])
        
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x    

# # 使用示例
# input_tensor = torch.randn(32, 64, 32, 32)  # 假设输入张量具有形状 [batch_size, channels, height, width]
# spatial_attention = SpatialAttention()
# output_tensor = spatial_attention(input_tensor)


class UNetConvNeXtV2Tiny(nn.Module):
    def __init__(self, in_chans=3, num_classes=4):
        super().__init__()
        self.encoder = convnextv2_tiny(in_chans=in_chans, num_classes=None,include_head=False)

        # 使用 ConvNeXtV2 Tiny 的第一个阶段作为瓶颈层
        #self.bottleneck = self.encoder.stages[0]
        # 修改bottleneck层为深度为3的ConvNeXtV2
        #使用三个block模块作为瓶颈层，每个block都是包括卷积，layer归一化和gelu激活函数
        self.bottleneck = nn.Sequential(
            *[Block(dim=1536, drop_path=0.) for _ in range(3)]
        )

        self.down_lay = nn.Sequential(
                LayerNorm(768, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(768, 1536, kernel_size=2, stride=2),
        )
        # self.down_lay = nn.Sequential( 
        #         nn.Conv2d(768, 1536, kernel_size=2, stride=2),
        #         LayerNorm(1536, eps=1e-6, data_format="channels_first"),
        # )

        
        self.block1 = Block(dim=1536, drop_path=0.)
        self.block2 = Block(dim=768, drop_path=0.)
        self.block3 = Block(dim=384, drop_path=0.)
        self.block4 = Block(dim=192, drop_path=0.)
    

        # 解码器部分
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1536, 768, kernel_size=3, padding=1),
                # Permute(0, 2, 3, 1),
                # LayerNorm(768, eps=1e-6),
                # Permute(0, 3, 1, 2),
                #
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(768),
                # nn.Conv2d(768, 768, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1536, 384, kernel_size=3, padding=1),
                # Permute(0, 2, 3, 1),
                # LayerNorm(384, eps=1e-6),
                # Permute(0, 3, 1, 2),
                #
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(384),
                # nn.Conv2d(384, 384, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(768, 192, kernel_size=3, padding=1),
                # Permute(0, 2, 3, 1),
                # nn.LayerNorm(192, eps=1e-6,),
                # Permute(0, 3, 1, 2),
                #
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(192),
                # nn.Conv2d(192, 192, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 96, kernel_size=3, padding=1),
                # Permute(0, 2, 3, 1),
                # nn.LayerNorm(96, eps=1e-6),
                # Permute(0, 3, 1, 2),
                #
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(96),
                # nn.Conv2d(96, 96, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                # Permute(0, 2, 3, 1),
                # nn.LayerNorm(96, eps=1e-6),
                # Permute(0, 3, 1, 2),
                #
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(96),
                # nn.Conv2d(96,96, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            #####是否可以换成gelu
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])
        
        """之前尝试的解码器卷积方法"""
        # self.res = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(1536, 768, kernel_size=1, stride=1),
        #         nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
        #         #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
        #         # nn.LayerNorm(768, eps=1e-6),
        #         # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
        #         nn.GELU()
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(1536, 384, kernel_size=1, stride=1),
        #         nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2),
        #         #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
        #         # nn.LayerNorm(384, eps=1e-6),
        #         # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)

        #         nn.GELU()
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(768, 192, kernel_size=1, stride=1),
        #         nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2),  # Add a transpose convolution layer to double the feature map siz
        #         #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
        #         # nn.LayerNorm(192, eps=1e-6),
        #         # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)

        #         nn.GELU()
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(384, 96, kernel_size=1, stride=1),
        #         nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2),  # Add a transpose convolution layer to double the feature map siz
        #         #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
        #         # nn.LayerNorm(96, eps=1e-6),
        #         # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)

        #         nn.GELU()
        #     ),
        #     #nn.Sequential(
        #     #    nn.Conv2d(256, 128, kernel_size=1, stride=1),
        #     #    nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)  # Add a transpose convolution layer to double the feature map siz
        #     #),
        #     #nn.Sequential(
        #     #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
        #     #    nn.ReLU(inplace=True),
        #     #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
        #     #    nn.ReLU(inplace=True)
        #     #)
        # ])



        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)
        """kanqingkuangshanchu"""
        self.final_conv1 = nn.Conv2d(1536, num_classes, kernel_size=1)
        self.final_conv2 = nn.Conv2d(768, num_classes, kernel_size=1)
        self.final_conv3 = nn.Conv2d(384, num_classes, kernel_size=1)
        self.final_conv4 = nn.Conv2d(192, num_classes, kernel_size=1)

        # self.grn1 = GRN(768)
        # self.grn2 = GRN(384)
        # self.grn3 = GRN(192)
        # self.grn4 = GRN(96)
        # self.grn5 = GRN(96)

        # 新增一个1x1卷积层，用于将第二个跨越连接的通道数增加两倍
        self.expand_skip_connection = nn.Conv2d(384, 768, kernel_size=1)
        # 新增一个1x1卷积层，用于将第二个跨越连接的通道数减少为原来的二分之一
        self.reduce_channel_conv = nn.Conv2d(192, 96, kernel_size=1)
        self.dwconv1 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, groups=384, bias=False)
        self.dwconv2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, groups=96, bias=False)
        self.expand_skip_connection2 = nn.Conv2d(96, 192, kernel_size=1)
        self.reduce_channel_conv1 = nn.Conv2d(384, 192, kernel_size=1)
        self.dwconv3 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, groups=192, bias=False)
        self.expand_skip_connection3 = nn.Conv2d(192, 384, kernel_size=1)
        self.reduce_channel_conv2 = nn.Conv2d(768, 384, kernel_size=1)

        #self.reduce_channel1 = nn.Conv2d(192, 96, kernel_size=1)
        #self.reduce_channel2 = nn.Conv2d(384, 192, kernel_size=1)
        self.reduce_channel1 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        #self.reduce_channel3 = nn.Conv2d(768, 384, kernel_size=1)
        self.reduce_channel2 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        #self.reduce_channel4 = nn.Conv2d(1536, 768, kernel_size=1)  #
        self.reduce_channel3 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        self.reduce_channel4 = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        # channel_attention1 = ChannelAttention(96)
        # channel_attention2 = ChannelAttention(192)
        # channel_attention3 = ChannelAttention(384)
        # channel_attention4 = ChannelAttention(768)

        # self.relu = nn.ReLU(inplace=True)

        # self.up1 = nn.Sequential(
        #     nn.Conv2d(1536, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(384, 384, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True)
        #     )
        # self.up2 = nn.Sequential(
        #     nn.Conv2d(384, 192, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(192, 192, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True)
        #     )
        # self.up3 = nn.Sequential(
        #     nn.Conv2d(192, 96, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(96, 96, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True)
        #     )
        # self.up4 = nn.Sequential(
        #     nn.Conv2d(1536, 96, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(96, 96, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True)
        #     )
        

        self.swin_layers = nn.ModuleList()
        self.swint_layers = nn.ModuleList()

        embed_dim = 96
        self.num_layers = 4
        self.image_size = 512
        #depths=[2, 2, 2, 2]
        depths=[2, 2, 2, 2]

        # num_heads=[2, 4, 8, 16]
        num_heads=[3, 6, 12, 24]
        window_size = 8 #先设置成8试一下下               #self.image_size// 16
        self.mlp_ratio = 4#.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size//4,self.image_size//4]  #解释

        
        #patch_size=[2, 4, 8, 16]

        for i_layer in range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)
        
        for i_layer in range(self.num_layers):
            swint_layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer+1)),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swint_layers.append(swint_layer)
         
        # """瓶颈层尝试加入swin_transformer"""
        # swin_layer5 = BasicLayer(dim=int(embed_dim * 2 ** 4),
        #                     input_resolution=(patches_resolution[0] // (2 ** 4),patches_resolution[1] // (2 ** 4)),
        #                     depth=depths[2],num_heads=num_heads[3],window_size=window_size,mlp_ratio=self.mlp_ratio,
        #                     qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
        #                     drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
        #                     norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
        # self.swin_layers.append(swin_layer5)

        """用于对比损失函数的部分函数"""
        """
        self.upa13 = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(768, 768, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )
        
        self.upa14 = nn.Sequential(
            nn.Conv2d(1536, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )
        
        self.upa15 = nn.Sequential(
            nn.Conv2d(1536, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            # nn.Conv2d(192, 192, kernel_size,=3, padding=1),
            # nn.ReLU(inplace=True)
            )

        self.upa23 = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(768, 768, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )

        self.upa24 = nn.Sequential(
            nn.Conv2d(1536, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )

        self.upa25 = nn.Sequential(
            nn.Conv2d(1536, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )

        self.upa34 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )
        self.upa35 = nn.Sequential(
            nn.Conv2d(768, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )
        self.upa45 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
            )
        """
        self.spatt1 = SpatialAttention1()
        self.channel_attention1 = ChannelAttention(192)
        self.channel_attention2 = ChannelAttention(384)
        self.channel_attention3 = ChannelAttention(768)
        self.channel_attention4 = ChannelAttention(1536)
        self.channel_attention5 = ChannelAttention(96)
        # self.channel_attention1 = ChannelAttention(96)
        # self.channel_attention2 = ChannelAttention(192)
        # self.channel_attention3 = ChannelAttention(384)
        # self.channel_attention4 = ChannelAttention(768)
        self.spatt2=ModifiedSpatialAttention()

        # self.channel1 = ModifiedChannelAttention(96)
        # self.channel2 = ModifiedChannelAttention(192)
        # self.channel3 = ModifiedChannelAttention(384)
        # self.channel4 = ModifiedChannelAttention(768)
        # self.channels = [
        #     ChannelAttention(96).to('cuda:0'),
        #     ChannelAttention(192).to('cuda:0'),
        #     ChannelAttention(384).to('cuda:0'),
        #     ChannelAttention(768).to('cuda:0')
        # ]
        self.channels = [
            # ModifiedChannelAttention(96).to('cuda:0'),
            # ModifiedChannelAttention(192).to('cuda:0'),
            # ModifiedChannelAttention(384).to('cuda:0'),
            # ModifiedChannelAttention(768).to('cuda:0')


            ModifiedChannelAttention(96).to('cpu'),
            ModifiedChannelAttention(192).to('cpu'),
            ModifiedChannelAttention(384).to('cpu'),
            ModifiedChannelAttention(768).to('cpu')

        ]

        ###### a1 1536 8  a2 1536 16 a3 768 32 a4 384 64 a5 192 128
        #a12 a13 a14 a15
        #a23 a24 a25
        #a34 a35
        #a45
        # channel_attention1 = ChannelAttention(192).to('cuda:0')#.to('cpu')
        # channel_attention2 = ChannelAttention(384).to('cuda:0')#.to('cpu')
        # channel_attention3 = ChannelAttention(768).to('cuda:0')#.to('cpu')
        # channel_attention4 = ChannelAttention(1536).to('cuda:0')#.to('cpu')

        """
        self.conv11 = nn.Conv2d(96, 24, kernel_size=5, padding=2)
        self.conv12 = nn.Conv2d(96, 24, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(192, 24, kernel_size=5, padding=2)
        self.conv14 = nn.Conv2d(192, 24, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(384, 192, kernel_size=5, padding=2)
        self.conv42 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(768, 192,kernel_size=5, padding=2)
        self.conv44 = nn.Conv2d(768, 192, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(96, 48, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(192, 48, kernel_size=5, padding=2)
        self.conv24 = nn.Conv2d(192, 48, kernel_size=3, padding=1)
        self.conv25 = nn.Conv2d(384, 48, kernel_size=5, padding=2)
        self.conv26 = nn.Conv2d(384, 48, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(192, 96, kernel_size=5, padding=2)
        self.conv32 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(384, 96, kernel_size=5, padding=2)
        self.conv34 = nn.Conv2d(384, 96, kernel_size=3, padding=1)
        self.conv35 = nn.Conv2d(768, 96, kernel_size=5, padding=2)
        self.conv36 = nn.Conv2d(768, 96, kernel_size=3, padding=1)
        """
        """原来的最红版本"""
        """
        self.conv11 = nn.Conv2d(96, 48, kernel_size=5, padding=2)
        self.conv12 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(192, 48, kernel_size=5, padding=2)
        self.conv14 = nn.Conv2d(192, 48, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(384, 384, kernel_size=5, padding=2)
        self.conv42 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(768, 384,kernel_size=5, padding=2)
        self.conv44 = nn.Conv2d(768, 384, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(96, 96, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(192, 96, kernel_size=5, padding=2)
        self.conv24 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.conv25 = nn.Conv2d(384, 96, kernel_size=5, padding=2)
        self.conv26 = nn.Conv2d(384, 96, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(192, 192, kernel_size=5, padding=2)
        self.conv32 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(384, 192, kernel_size=5, padding=2)
        self.conv34 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
        self.conv35 = nn.Conv2d(768, 192, kernel_size=5, padding=2)
        self.conv36 = nn.Conv2d(768, 192, kernel_size=3, padding=1)
        """
        # self.conv11 = nn.Conv2d(96, 96, kernel_size=5, padding=2)
        # self.conv12 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        # self.conv13 = nn.Conv2d(192, 96, kernel_size=5, padding=2)
        # self.conv14 = nn.Conv2d(192, 96, kernel_size=3, padding=1)

        # self.conv41 = nn.Conv2d(384, 768, kernel_size=5, padding=2)
        # self.conv42 = nn.Conv2d(384, 768, kernel_size=3, padding=1)
        # self.conv43 = nn.Conv2d(768, 768,kernel_size=5, padding=2)
        # self.conv44 = nn.Conv2d(768, 768, kernel_size=3, padding=1)

        # self.conv21 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        # self.conv22 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        # self.conv23 = nn.Conv2d(192, 192, kernel_size=5, padding=2)
        # self.conv24 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        # self.conv25 = nn.Conv2d(384, 192, kernel_size=5, padding=2)
        # self.conv26 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # self.conv31 = nn.Conv2d(192, 384, kernel_size=5, padding=2)
        # self.conv32 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # self.conv33 = nn.Conv2d(384, 384, kernel_size=5, padding=2)
        # self.conv34 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        # self.conv35 = nn.Conv2d(768, 384, kernel_size=5, padding=2)
        # self.conv36 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        
        """该用1和3的卷积核"""
        self.conv11 = nn.Conv2d(96,48, kernel_size=1)
        self.conv12 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(192, 48, kernel_size=1)
        self.conv14 = nn.Conv2d(192, 48, kernel_size=3, padding=1)

        self.conv41 = nn.Conv2d(384, 384, kernel_size=1)
        self.conv42 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(768,384,kernel_size=1)
        self.conv44 = nn.Conv2d(768,384, kernel_size=3, padding=1)

        self.conv21 = nn.Conv2d(96, 96, kernel_size=1)
        self.conv22 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(192, 96, kernel_size=1)
        self.conv24 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.conv25 = nn.Conv2d(384, 96, kernel_size=1)
        self.conv26 = nn.Conv2d(384, 96, kernel_size=3, padding=1)

        self.conv31 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv32 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(384, 192, kernel_size=1)
        self.conv34 = nn.Conv2d(384,192, kernel_size=3, padding=1)
        self.conv35 = nn.Conv2d(768, 192, kernel_size=1)
        self.conv36 = nn.Conv2d(768, 192, kernel_size=3, padding=1)


        
    # def FF_2(max,me,min):
    #     # self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    #     # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    #     # self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
    #     x = 
    #     reture x

    def forward(self, x):
        skip_connections = []
        skip_convnext=[]
        #print("Before bottleneck:", x.shape)
        # 编码器部分
        for i in range(4):
            x = self.encoder.downsample_layers[i](x)   ###下面两个换一下顺序
            # # u = x                               ###用了残差加入空间注意力      
            x = self.encoder.stages[i](x)
            #这里从convnext单独引出一个特征，用以恢复多尺度特征
            skip_convnext.append(x)
            
            u = x
            # #x = self.spatt1(x) #TIA添加空间注意力在每个stage之后，下采样之前     
            x = self.spatt2(x)
            # x = self.channels[i](x)
            x = x + u
            # u=x
            # u=self.spatt1(u)
            
            x = x.flatten(2).transpose(1, 2) 
            # print("Before bottleneck:", x.shape) 8,16384,96 8,4096,192 8,1024,384 8,256,768
            x = self.swin_layers[i] (x)
            B, L, C = x.shape
            x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
            #x = x + u
            #x = self.spatt1(x) #TIA添加空间注意力在每个stage之后，下采样之前
            
            skip_connections.append(x)
            #print("Before bottleneck:", x.shape)
        #print("Before bottleneck:", x.shape)
        # 使用瓶颈层处理特征        

        

        """在这里加入vit或swin进行特征提取"""
        #z1 = self.vit(z1)
        # h1 = skip_connections[0].flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 

        # h1 = self.swin_layers[0] (h1)
        
        # B, L, C = h1.shape

        # h1 = h1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        # h2 = skip_connections[1].flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128
        # h2 = self.swin_layers[1] (h2)
        # B, L, C = h2.shape
        # h2 = h2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        # h3 = skip_connections[2].flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128
        # h3 = self.swin_layers[2] (h3)
        # B, L, C = h3.shape
        # h3 = h3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        # h4 = skip_connections[3].flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128
        # h4 = self.swin_layers[3] (h4)
        # B, L, C = h4.shape
        # h4 = h4.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        # skip_connections[0] = h1
        # skip_connections[1] = h2
        # skip_connections[2] = h3
        # skip_connections[3] = h4


        ###768 12    384 24   192 48   96 96
        """
        # 处理跨越连接
        # 对第二个跨越连接首先进行1*1卷积使通道数减少为原来的二分之一
        skip_connection_1 = self.reduce_channel_conv(skip_connections[1])
        # 对第二个跨越连接进行双线性插值上采样使得特征图尺寸扩大两倍
        skip_connection_1 = F.interpolate(skip_connection_1, scale_factor=2, mode='bilinear', align_corners=False)
        # 让第一个跨越连接与第二个处理后的跨越连接进行逐元素相乘，得到新的第一个跨越连接
        new_skip_connection_0 = skip_connections[0] + skip_connection_1   ##加号变成乘号就可以
        # 使用新的第一个跨越连接替换原始跨越连接列表中的第一个跨越连接
        #skip_connections[0] = new_skip_connection_0

        #处理第三个跨越连接的特征
        skip_connection_3 = self.dwconv1(skip_connections[2])
        skip_connection_3 = self.expand_skip_connection(skip_connection_3)
        new_skip_connection_4 = skip_connections[3] + skip_connection_3
        #skip_connections[3] = new_skip_connection_4


       #处理第二个跨越连接的特征
        skip_connection_21 = self.dwconv2(skip_connections[0])
        skip_connection_21 = self.expand_skip_connection2(skip_connection_21)   #将第一个跨层连接降采样
        new_skip_connection_21 = skip_connections[1] + skip_connection_21

        skip_connection_23 = self.reduce_channel_conv1(skip_connections[2]) 
        skip_connection_23 = F.interpolate(skip_connection_23, scale_factor=2, mode='bilinear', align_corners=False)
        new_skip_connection_23 = skip_connections[1] + skip_connection_23

        #skip_connections[1] = self.reduce_channel_conv1(skip_connections[1])

        #处理第三个跨越连接的数据
        skip_connection_32 = self.dwconv3(skip_connections[1])
        skip_connection_32 = self.expand_skip_connection3(skip_connection_32)   #将第一个跨层连接降采样
        new_skip_connection_32 = skip_connections[2] + skip_connection_32

        skip_connection_34 = self.reduce_channel_conv2(skip_connections[3]) 
        skip_connection_34 = F.interpolate(skip_connection_34, scale_factor=2, mode='bilinear', align_corners=False)
        new_skip_connection_34 = skip_connections[2] + skip_connection_34

        #skip_connections[2] = self.reduce_channel_conv2(skip_connections[2])
        """

        
        z1 = skip_connections[0]
        z2 = skip_connections[1]
        z3 = skip_connections[2]
        z4 = skip_connections[3]
       
        # 对第二个跨越连接进行双线性插值上采样使得特征图尺寸扩大两倍
        skip_connection_1 = F.interpolate(skip_connections[1], scale_factor=2, mode='bilinear', align_corners=False)
        d11 = self.conv11(skip_connections[0])
        d12 = self.conv12(skip_connections[0])
        """处理convnext引出的特征"""
        dnext11 = self.conv11(skip_convnext[0])
        dnext12 = self.conv12(skip_convnext[0])
        d11 = d11 + dnext11
        d12 = d12 + dnext12
        """"""
        d13 = self.conv13(skip_connection_1)
        d14 = self.conv14(skip_connection_1)
        d1 = torch.cat((d11,d12),dim=1)
        d2 = torch.cat((d13,d14),dim=1)
        # d1 = d11+d12
        # d2 = d13+d14

        new_skip_connection_0 = d1+d2
        #new_skip_connection_0 = torch.cat((d11,d12,d13,d14),dim=1)
 
        #处理第三个跨越连接的特征,特诊图减小两倍
        skip_connection_3 = self.dwconv1(skip_connections[2])
        d41 = self.conv41(skip_connection_3)
        d42 = self.conv42(skip_connection_3)
        d43 = self.conv43(skip_connections[3])
        d44 = self.conv44(skip_connections[3])
        """"""
        dnext43 = self.conv43(skip_convnext[3])
        dnext44 = self.conv44(skip_convnext[3])
        d43 = d43 + dnext43
        d44 = d44 + dnext44
        """"""
        d3 = torch.cat((d43,d44),dim=1)
        d4 = torch.cat((d41,d42),dim=1)
        # d3 = d43 +d44
        # d4 = d41 +d42

        new_skip_connection_4 = d3+d4
        #new_skip_connection_4 = torch.cat((d41,d42,d43,d44),dim=1)

       #处理第二个跨越连接的特征
        skip_connection_21 = self.dwconv2(skip_connections[0])
        d21 = self.conv21(skip_connection_21)
        d22 = self.conv22(skip_connection_21)
        d23 = self.conv23(skip_connections[1])
        d24 = self.conv24(skip_connections[1])
        """"""
        dnext23 = self.conv23(skip_convnext[1])
        dnext24 = self.conv24(skip_convnext[1])
        d23 = d23 + dnext23
        d24 = d24 + dnext24
        """"""
        d5 = torch.cat((d23,d24),dim=1)
        d6 = torch.cat((d21,d22),dim=1)
        # d5 = d23 + d24
        # d6 = d21 + d22

        new_skip_connection_21 = d5+d6
        #new_skip_connection_21= torch.cat((d21,d22,d23,d24),dim=1)
        
        skip_connection_23 = F.interpolate(skip_connections[2], scale_factor=2, mode='bilinear', align_corners=False)
        d25 = self.conv25(skip_connection_23)
        d26 = self.conv26(skip_connection_23)
        d7 = torch.cat((d25,d26),dim=1)
        # d7 = d25 +d26

        new_skip_connection_23 = d5 +d7
        #new_skip_connection_23= torch.cat((d23,d24,d25,d26),dim=1)

        #处理第三个跨越连接的数据
        skip_connection_32 = self.dwconv3(skip_connections[1])
        d31 = self.conv31(skip_connection_32)
        d32 = self.conv32(skip_connection_32)
        d33 = self.conv33(skip_connections[2])
        d34 = self.conv34(skip_connections[2])
        """"""
        dnext33 = self.conv33(skip_convnext[2])
        dnext34 = self.conv34(skip_convnext[2])
        d33 = d33 + dnext33
        d34 = d34 + dnext34
        """"""
        d8= torch.cat((d31,d32),dim=1)
        d9 = torch.cat((d33,d34),dim=1)
        # d8 = d31+d32
        # d9 = d33+d34

        new_skip_connection_32= d8 + d9
        #new_skip_connection_32= torch.cat((d31,d32,d33,d34),dim=1)

        skip_connection_34 = F.interpolate(skip_connections[3], scale_factor=2, mode='bilinear', align_corners=False)
        d35 = self.conv35(skip_connection_34)
        d36 = self.conv36(skip_connection_34)
        d10 = torch.cat((d35,d36),dim=1)
        # d10 = d35+d36

        new_skip_connection_34= d9 +d10
        #new_skip_connection_34= torch.cat((d33,d34,d35,d36),dim=1)


        #zaizheli将通道注意力加上去
        new_skip_connection_0 = self.channel_attention5(new_skip_connection_0)
        new_skip_connection_4 = self.channel_attention3(new_skip_connection_4)
        new_skip_connection_21 = self.channel_attention1(new_skip_connection_21)
        new_skip_connection_23 = self.channel_attention1(new_skip_connection_23)
        new_skip_connection_32 = self.channel_attention2(new_skip_connection_32)
        new_skip_connection_34 = self.channel_attention2(new_skip_connection_34)

        ##### ""在这里啊面进行特征最后的融合""
        skip_connections[0] = torch.cat([new_skip_connection_0, skip_connections[0]], dim=1)# 384 12 
        skip_connections[3] = torch.cat([new_skip_connection_4, skip_connections[3]], dim=1)# 384 12 
        skip_connections[1] = torch.cat([new_skip_connection_21, new_skip_connection_23], dim=1)# 384 12 
        skip_connections[2] = torch.cat([new_skip_connection_32, new_skip_connection_34], dim=1)# 384 12 
       
        #加残差大小必须一致，所以必须加上这个内容，如果残差加在这里效果不好就算了
        ###""把z1的大小变大，算了还先变小试一下""
        # c1 = skip_connections[0]
        # c2 = skip_connections[1]
        # c3 = skip_connections[2]
        # c4 = skip_connections[3]

        # c1 = self.reduce_channel1(c1)
        # c2 = self.reduce_channel2(c2)
        # c3 = self.reduce_channel3(c3)
        # c4 = self.reduce_channel4(c4)


        # skip_connections[0] = z1 + c1
        # skip_connections[1] = z2 + c2
        # skip_connections[2] = z3 + c3
        # skip_connections[3] = z4 + c4


        # channel_attention1 = ChannelAttention(192).to('cuda:0')#.to('cpu')
        # channel_attention2 = ChannelAttention(384).to('cuda:0')#.to('cpu')
        # channel_attention3 = ChannelAttention(768).to('cuda:0')#.to('cpu')
        # channel_attention4 = ChannelAttention(1536).to('cuda:0')#.to('cpu')

        ### channel_skip_connection1 = self.channel_attention1(skip_connections[0])
        ### channel_skip_connection2 = self.channel_attention2(skip_connections[1])
        ### channel_skip_connection3 = self.channel_attention3(skip_connections[2])
        ### channel_skip_connection4 = self.channel_attention4(skip_connections[3])
        # skip_connections[0] = self.channel_attention1(skip_connections[0])
        # skip_connections[1] = self.channel_attention2(skip_connections[1])
        # skip_connections[2] = self.channel_attention3(skip_connections[2])
        # skip_connections[3] = self.channel_attention4(skip_connections[3])
        # print("channel_attention shape:",channel_skip_connection1.shape)
        # print("channel_attention shape:",channel_skip_connection2.shape)

        # spatial_attention = SpatialAttention().to('cuda')                     #.to('cuda:0')
        # spatial_skip_connection1 = spatial_attention(channel_skip_connection1)
        # spatial_skip_connection2 = spatial_attention(channel_skip_connection2)
        # spatial_skip_connection3 = spatial_attention(channel_skip_connection3)
        # spatial_skip_connection4 = spatial_attention(channel_skip_connection4)
        ### spatial_skip_connection1 = self.spatt1(channel_skip_connection1)
        ### spatial_skip_connection2 = self.spatt1(channel_skip_connection2)
        ### spatial_skip_connection3 = self.spatt1(channel_skip_connection3)
        ### spatial_skip_connection4 = self.spatt1(channel_skip_connection4)
        # print("spatial_attention shape:",spatial_skip_connection1.shape)
        # print("channel_attention shape:",spatial_skip_connection2.shape)
        # spatial_skip_connection2 = spatial_attention(channel_attention2)
        # spatial_skip_connection3 = spatial_attention(channel_attention3)
        # spatial_skip_connection4 = spatial_attention(channel_attention4)
        # spatial_skip_connection1 = spatial_attention(skip_connections[0])
        # spatial_skip_connection2 = spatial_attention(skip_connections[1])
        # spatial_skip_connection3 = spatial_attention(skip_connections[2])
        # spatial_skip_connection4 = spatial_attention(skip_connections[3])
        ### skip_connections[0] =  spatial_skip_connection1
        ### skip_connections[1] =  spatial_skip_connection2
        ### skip_connections[2] =  spatial_skip_connection3
        ### skip_connections[3] =  spatial_skip_connection4

        # skip_connections[0] =  (channel_skip_connection1 + spatial_skip_connection1)
        # skip_connections[1] =  (channel_skip_connection2 + spatial_skip_connection2)
        # skip_connections[2] =  (channel_skip_connection3 + spatial_skip_connection3)
        # skip_connections[3] =  (channel_skip_connection4 + spatial_skip_connection4)
        # print("skip_connections[0] shape:",skip_connections[0].shape)

        ###""通道数在上面就变好了""
        skip_connections[0] = self.reduce_channel1(skip_connections[0])
        skip_connections[1] = self.reduce_channel2(skip_connections[1])
        skip_connections[2] = self.reduce_channel3(skip_connections[2])
        skip_connections[3] = self.reduce_channel4(skip_connections[3])

        ###""去掉这几个部分，最原始的作用,relu本来就是去掉的、要是不加这个特征融合会不会的结果会更好呢？""
        # skip_connections[0] = self.relu(skip_connections[0])
        # skip_connections[1] = self.relu(skip_connections[1])
        # skip_connections[2] = self.relu(skip_connections[2])

        # skip_connections[3] = self.relu(skip_connections[3])
        
        skip_connections[0] = z1 + skip_connections[0]
        skip_connections[1] = z2 + skip_connections[1]
        skip_connections[2] = z3 + skip_connections[2]
        skip_connections[3] = z4 + skip_connections[3]
        #写一段python代码，实现skip_connections[0]的batchnorm归一化

        # skip_connections[0] = self.bn1(skip_connections[0])
        # self.bn1 = nn.BatchNorm2d(384)   
        # skip_connections3通道数
        
        

        x = self.down_lay(x) #1536 6  下采样

        """瓶颈层换用swin"""
        x = self.bottleneck(x)  #1536 6
        # x = x.flatten(2).transpose(1, 2) 
        # x = self.swin_layers[4] (x)
        # B, L, C = x.shape
        # x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        # y1 = x #1536 6
        a1 = x
        #print("After bottleneck:", x.shape)
        # 解码器部分
        #for i in range(3):
        #    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #    x = torch.cat([x, skip_connections[2 - i]], dim=1)
        #    x = self.decoder[i](x)
        ###resc1 = self.res[0](x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        x = self.decoder[0](x)  #768 12
        # x = x.permute(0, 2, 3, 1)
        # x = self.grn1(x)  ###########加入全局响应归一化层
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc1


        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 

        
        ###resc2 = self.res[1](x)
        ######x = self.block1(x) #1536
        
        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[3] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        a2 = x
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        x = self.decoder[1](x)  #384 24
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.grn2(x)  ###########加入全局响应归一化层
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc2
        
        #print("Before bottleneck:", x.shape)
        # y2 = x #384 24

        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24

        
        ###resc3 = self.res[2](x)
        ######x = self.block2(x)  #768
        
        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[2] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        a3 = x
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        x = self.decoder[2](x) #192 48
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.grn3(x)  ###########加入全局响应归一化层
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc3
        
        # y3 = x #192 48
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        
        ###resc4 = self.res[3](x)
        ######x = self.block3(x) #384
        
        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[1] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        a4 = x 
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        x = self.decoder[3](x)  # 96 96
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.grn4(x)  ###########加入全局响应归一化层
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc4
        
        # y4 = x #96 96
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        
        ######x= self.block4(x) #192
        
        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[0] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)
        a5 = x
        
        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.grn5(x)  ###########加入全局响应归一化层
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
        #x = torch.cat([x, skip_connections[0][:, :, :x.shape[2], :x.shape[3]]], dim=1)
        #x = self.decoder[3](x)
        #print("After bottleneck:", x.shape)

        x = self.final_conv(x)  #4 384 384 
        #print("Before bottleneck:", x.shape)
        """
        m1 = F.interpolate(y1, scale_factor=4, mode='bilinear', align_corners=False) #1536 24
        m1 = self.up1(m1) #4 384 24
        m2 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=False) #384 48
        m2 = self.up2(m2) #4 192 48
        m3 = F.interpolate(y3, scale_factor=2, mode='bilinear', align_corners=False) #192 96
        m3 = self.up3(m3) #4 96 96
        #y4 = y4 #96 96
        
        #m1,y2   m2,y3   m3,y4
        #对m1进行softmax
        n1 = m1.permute(0, 2, 3, 1)
        n1 = torch.nn.functional.softmax(n1, dim=3)
        n1 = n1.permute(0, 3, 1, 2)
        #  n1的shape是
        n2 = y2.permute(0, 2, 3, 1)
        n2 = torch.nn.functional.softmax(n2, dim=3)
        n2 = n2.permute(0, 3, 1, 2)
        #  n2的shape是
        n3 = m2.permute(0, 2, 3, 1)
        n3 = torch.nn.functional.softmax(n3, dim=3)
        n3 = n3.permute(0, 3, 1, 2)
        #  n3的shape是
        n4 = y3.permute(0, 2, 3, 1)
        n4 = torch.nn.functional.softmax(n4, dim=3)
        n4 = n4.permute(0, 3, 1, 2)
        #  n4的shape是
        n5 = m3.permute(0, 2, 3, 1)
        n5 = torch.nn.functional.softmax(n5, dim=3)
        n5 = n5.permute(0, 3, 1, 2)
        #  n5的shape是
        n6 = y4.permute(0, 2, 3, 1)
        n6 = torch.nn.functional.softmax(n6, dim=3)
        n6 = n6.permute(0, 3, 1, 2)
        #  n6的shape是

        #加一个y1和y4的对比
        #y1大小1536 6
        #y4大小96 96
        m5 = F.interpolate(y1, scale_factor=16, mode='bilinear', align_corners=False) #1536 96
        m5 = self.up4(m5) #96 96
        n7 = m5.permute(0, 2, 3, 1)
        n7 = torch.nn.functional.softmax(n7, dim=3)
        n7 = n7.permute(0, 3, 1, 2)
        
        ###### a1 1536 8  a2 1536 16 a3 768 32 a4 384 64 a5 192 128
        #a12 a13 a14 a15
        #a23 a24 a25
        #a34 a35
        #a45
        """

        """最后的用于对比损失函数的部分，肯定是不用了，跑的太慢了，最后连结果都跑不出来"""
        """
        a12 = F.interpolate(a1, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        a13 = F.interpolate(a1, scale_factor=4, mode='bilinear', align_corners=False)
        a14 = F.interpolate(a1, scale_factor=8, mode='bilinear', align_corners=False)
        a15 = F.interpolate(a1, scale_factor=16, mode='bilinear', align_corners=False)
        
        a13 = self.upa13(a13)
        a14 = self.upa14(a14)
        a15 = self.upa15(a15)

        a23 = F.interpolate(a2, scale_factor=2, mode='bilinear', align_corners=False)
        a24 = F.interpolate(a2, scale_factor=4, mode='bilinear', align_corners=False)
        a25 = F.interpolate(a2, scale_factor=8, mode='bilinear', align_corners=False)
        a23 = self.upa23(a23)
        a24 = self.upa24(a24)
        a25 = self.upa25(a25)

        a34 = F.interpolate(a3, scale_factor=2, mode='bilinear', align_corners=False)
        a35 = F.interpolate(a3, scale_factor=4, mode='bilinear', align_corners=False)
        a34 = self.upa34(a34)
        a35 = self.upa35(a35)

        a45 = F.interpolate(a4, scale_factor=2, mode='bilinear', align_corners=False)
        a45 = self.upa45(a45)
        

        a12 = self.final_conv1(a12)
        a2 =  self.final_conv1(a2)
        a23 = self.final_conv2(a23)
        a3 = self.final_conv2(a3)
        a34 = self.final_conv3(a34)
        a4 = self.final_conv3(a4)
        a45 = self.final_conv4(a45)
        a5 = self.final_conv4(a5)
        """
        # a12 = a12.permute(0, 2, 3, 1)
        # a12 = torch.nn.functional.softmax(a12, dim=3)
        # a12 = a12.permute(0, 3, 1, 2)
        
        # a13 = a13.permute(0, 2, 3, 1)
        # a13 = torch.nn.functional.softmax(a13, dim=3)
        # a13 = a13.permute(0, 3, 1, 2)
        
        # a14 = a14.permute(0, 2, 3, 1)
        # a14 = torch.nn.functional.softmax(a14, dim=3)
        # a14 = a14.permute(0, 3, 1, 2)
        
        # a15 = a15.permute(0, 2, 3, 1)
        # a15 = torch.nn.functional.softmax(a15, dim=3)
        # a15 = a15.permute(0, 3, 1, 2)

        # a23 = a23.permute(0, 2, 3, 1)
        # a23 = torch.nn.functional.softmax(a23, dim=3)
        # a23 = a23.permute(0, 3, 1, 2)
        
        # a24 = a24.permute(0, 2, 3, 1)
        # a24 = torch.nn.functional.softmax(a24, dim=3)
        # a24 = a24.permute(0, 3, 1, 2)
        
        # a25 = a25.permute(0, 2, 3, 1)
        # a25 = torch.nn.functional.softmax(a25, dim=3)
        # a25 = a25.permute(0, 3, 1, 2)
        
        # a34 = a34.permute(0, 2, 3, 1)
        # a34 = torch.nn.functional.softmax(a34, dim=3)
        # a34 = a34.permute(0, 3, 1, 2)
        
        # a35 = a35.permute(0, 2, 3, 1)
        # a35 = torch.nn.functional.softmax(a35, dim=3)
        # a35 = a35.permute(0, 3, 1, 2)

        # a45 = a45.permute(0, 2, 3, 1)
        # a45 = torch.nn.functional.softmax(a45, dim=3)
        # a45 = a45.permute(0, 3, 1, 2)

        #a1 = a1.view(batch_siz  , channels, -1)
        
        #return x , m1 , y2 , m2 , y3 , m3 ,y4, n1,n2,n3,n4,n5,n5,n6 ,n7   #,y1,y2,y3,y4
        return x ###,a12,a2,a23,a3,a34,a4,a45,a5######, a12,a2, a13,a3,a14,a4,a15,a5,a23,a3,a24,a4,a25,a5,a34,a4,a35,a5,a45,a5
        #按照12345，第一个next后面，其他都是swin后面。
        #n1等都是在经过softmax的结果
        # 去掉12 910 1516
        #请对上面的这个class的代码进行解释，尤其是forward函数的代码，解释每一行代码的作用，以及每一行代码的输入和输出的shape
        # 请在下面的代码中写出你的答案
        
    

        

    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.encoder.state_dict()
            #model_dict = self.state_dict()


            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict = {k.replace("head.", ""): v for k, v in pretrained_dict.items() if k.startswith("head.") or k in model_dict}
            #model_dict.update(pretrained_dict)
            #model.load_state_dict(pretrained_dict,)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            #msg = model.load_state_dict(pretrained_dict, strict=False)
            #print("missing keys:", msg.missing_keys)

        else:
            print("No pretrained model provided, training from scratch.")
    
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.encoder.state_dict()

            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict = {k.replace("head.", ""): v for k, v in pretrained_dict.items() if k.startswith("head.") or k in model_dict}
            print(f"Number of pretrained weights: {len(pretrained_dict)}")
        
            with open("output111.txt", "w") as file:
                with contextlib.redirect_stdout(file):
                    # Print weights before loading
                    print("Model weights before loading pretrained weights:")
                    for k, v in model_dict.items():
                        print(f"{k}: {v}")

                    # Update model weights with pretrained weights
                    model_dict.update(pretrained_dict)
                    self.load_state_dict(model_dict, strict=False)

                    # Print weights after loading
                    print("Model weights after loading pretrained weights:")
                    for k, v in self.encoder.state_dict().items():
                        print(f"{k}: {v}")
        else:
            print("No pretrained model provided, training from scratch.")
    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.state_dict()

            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict_e = {}
            for k, v in pretrained_dict['model'].items():
                if 'stages' in k:
                    pretrained_dict_e[k.replace('stages', 'encoder.stages')] = v
                elif 'downsample' in k:
                    pretrained_dict_e[k.replace('downsample_layers', 'encoder.downsample_layers')] = v
                elif 'norm' in k:
                    pretrained_dict_e[k.replace('norm', 'encoder.norm')] = v

            for key, value in pretrained_dict_e.items():
                if key in model_dict.keys():
                    model_dict[key] = value

            self.load_state_dict(model_dict, strict=False)
        else:
            print("No pretrained model provided, training from scratch.")



###加载预训练模型时出现了问题，模型参数没有加载进去，load代码应该如何修改。
# 创建U-Net模型实例
#model = UNetConvNeXtV2Tiny()


class UNetConvNeXtV2Pico(nn.Module):
    def __init__(self, in_chans=3, num_classes=4):
        super().__init__()
        self.encoder = convnext_pico(in_chans=in_chans, num_classes=None,include_head=False)

        # 使用 ConvNeXtV2 Tiny 的第一个阶段作为瓶颈层
        #self.bottleneck = self.encoder.stages[0]
        # 修改bottleneck层为深度为3的ConvNeXtV2
        self.bottleneck = nn.Sequential(
            *[Block(dim=1024, drop_path=0.) for _ in range(3)]
        )

        self.down_lay = nn.Sequential(
                LayerNorm(512, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(512, 1024, kernel_size=2, stride=2),
        )

        # 解码器部分
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])



        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        #print("Before bottleneck:", x.shape)
        # 编码器部分
        for i in range(4):
            x = self.encoder.downsample_layers[i](x)   ###下面两个换一下顺序
            
            x = self.encoder.stages[i](x)
            skip_connections.append(x)
            #print("Before bottleneck:", x.shape)
        #print("Before bottleneck:", x.shape)
        # 使用瓶颈层处理特征
        
        ###768 12    384 24   192 48   96 96

        x = self.down_lay(x) #1536 6


        x = self.bottleneck(x)  #1536 6
        
        #print("After bottleneck:", x.shape)
        # 解码器部分
        #for i in range(3):
        #    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #    x = torch.cat([x, skip_connections[2 - i]], dim=1)
        #    x = self.decoder[i](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        x = self.decoder[0](x)  #768 12
        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        x = self.decoder[1](x)  #384 24

        
        #print("Before bottleneck:", x.shape)
        
        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        x = self.decoder[2](x) #192 48
        
        
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        x = self.decoder[3](x)  # 96 96
        
        
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)


        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
        #x = torch.cat([x, skip_connections[0][:, :, :x.shape[2], :x.shape[3]]], dim=1)
        #x = self.decoder[3](x)
        #print("After bottleneck:", x.shape)

        x = self.final_conv(x)  #4 384 384 
        #print("Before bottleneck:", x.shape)
        return x
    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            #model_dict = self.encoder.state_dict()
            model_dict = self.state_dict()


            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict = {k.replace("head.", ""): v for k, v in pretrained_dict.items() if k.startswith("head.") or k in model_dict}
            #model_dict.update(pretrained_dict)
            #model.load_state_dict(pretrained_dict,)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            #msg = model.load_state_dict(pretrained_dict, strict=False)
            #print("missing keys:", msg.missing_keys)

        else:
            print("No pretrained model provided, training from scratch.")
    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.state_dict()

            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict_e = {}
            for k, v in pretrained_dict['model'].items():
                if 'stages' in k:
                    pretrained_dict_e[k.replace('stages', 'encoder.stages')] = v
                elif 'downsample' in k:
                    pretrained_dict_e[k.replace('downsample_layers', 'encoder.downsample_layers')] = v
                elif 'norm' in k:
                    pretrained_dict_e[k.replace('norm', 'encoder.norm')] = v

            for key, value in pretrained_dict_e.items():
                if key in model_dict.keys():
                    model_dict[key] = value

            self.load_state_dict(model_dict, strict=False)
        else:
            print("No pretrained model provided, training from scratch.")
            

class Permute(nn.Module):
    def __init__(self, *dims):        
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class UNetConvNeXtV2Base(nn.Module):
    def __init__(self, in_chans=3, num_classes=4):
        super().__init__()
        self.encoder = convnextv2_base(in_chans=in_chans, num_classes=None,include_head=False)

        # 使用 ConvNeXtV2 Tiny 的第一个阶段作为瓶颈层
        #self.bottleneck = self.encoder.stages[0]
        # 修改bottleneck层为深度为3的ConvNeXtV2
        self.bottleneck = nn.Sequential(
            *[Block(dim=2048, drop_path=0.) for _ in range(3)]
        )

        self.down_lay = nn.Sequential(
                LayerNorm(1024, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(1024, 2048, kernel_size=2, stride=2),
        )

        # 解码器部分
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])
        """
        """
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.GELU(), 
                nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)  #解释这段代码 Add a transpose convolution layer to double the feature map siz

            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(128,128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)  # Add a transpose convolution layer to double the feature map siz
            ),
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])
        """
        
        self.res = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
                nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                #x.mean([-2,-1],)
                # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
                # nn.LayerNorm(1024, eps=1e-6),
                # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1),
                nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
                # nn.LayerNorm(512, eps=1e-6),
                # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # Add a transpose convolution layer to double the feature map siz
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
                # nn.LayerNorm(256, eps=1e-6),
                # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
                # nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, stride=1),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # Add a transpose convolution layer to double the feature map siz
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # Permute(0, 2, 3, 1), # (N, C, H, W) -> (N, H, W, C)
                # nn.LayerNorm(128, eps=1e-6),
                # Permute(0, 3, 1, 2), # (N, H, W, C) -> (N, C, H, W)
                nn.GELU()
            ),
            #nn.Sequential(
            #    nn.Conv2d(256, 128, kernel_size=1, stride=1),
            #    nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)  # Add a transpose convolution layer to double the feature map siz
            #),
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                #LayerNorm(2048, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                # nn.GELU(),
                nn.ReLU(inplace=True),
                # nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                # nn.GELU(),
                #nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                #LayerNorm(2048, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # nn.GELU(),
                # nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.GELU(),
                #nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                #LayerNorm(1024, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                # nn.GELU(),
                nn.ReLU(inplace=True),
                # nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # nn.GELU()
                #nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                #LayerNorm(512, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                # nn.GELU(),
                nn.ReLU(inplace=True),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.GELU()
                #nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                #LayerNorm(256, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # nn.Conv2d(128,128, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            ),
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])


        #如果是这样处理的话，则需要将最后的输出进行处理，将输出的通道数变为4最后又加了一个上采样的块，其实不加也就是可以的
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

        self.grn1 = GRN(1024)
        self.grn2 = GRN(512)
        self.grn3 = GRN(256)
        self.grn4 = GRN(128)
        self.grn5 = GRN(128)
        
        # self.bottleneck = nn.Sequential(
        #     *[Block(dim=2048, drop_path=0.) for _ in range(3)]
        # )

        self.block1 = nn.Sequential(
            *[Block(dim=2048, drop_path=0.) for _ in range(2)]
        )#Block(dim=2048, drop_path=0.)
        self.block2 = nn.Sequential(
            *[Block(dim=1024, drop_path=0.) for _ in range(2)]
        )# Block(dim=1024, drop_path=0.)
        self.block3 = nn.Sequential(
            *[Block(dim=512, drop_path=0.) for _ in range(2)]
        )#Block(dim=512, drop_path=0.)
        self.block4 = nn.Sequential(
            *[Block(dim=256, drop_path=0.) for _ in range(2)]
        )#Block(dim=256, drop_path=0.)


        """swin transformer的基本模块"""
        self.swin_layers = nn.ModuleList()
        self.swint_layers = nn.ModuleList()

        embed_dim = 128
        self.num_layers = 4
        self.image_size = 512
        depths=[2, 2, 2, 2]
        num_heads=[2, 4, 8, 16]
        window_size = 8 #先设置成8试一下下               #self.image_size// 16
        self.mlp_ratio = 4#.0
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patches_resolution = [self.image_size//4,self.image_size//4]  #解释

        
        #patch_size=[2, 4, 8, 16]

        for i_layer in range(self.num_layers):
            swin_layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swin_layers.append(swin_layer)

        for i_layer in range(self.num_layers):
            swint_layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer+1)),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                               qkv_bias=True, qk_scale=None,drop=0.0, attn_drop=0.0,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,downsample= None,use_checkpoint=False)
            self.swint_layers.append(swint_layer)

        self.spat1 = SpatialAttention() 


    def forward(self, x):
        skip_connections = []
        #print("Before bottleneck:", x.shape)
        # 编码器部分
        for i in range(4):
            x = self.encoder.downsample_layers[i](x)   ###下面两个换一下顺序
            
            x = self.encoder.stages[i](x)
            x = self.spat1(x)  
            x = x.flatten(2).transpose(1, 2) 
            # print("Before bottleneck:", x.shape) 8,16384,96 8,4096,192 8,1024,384 8,256,768
            x = self.swin_layers[i] (x)
            B, L, C = x.shape
            x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

            skip_connections.append(x)
            #print("Before bottleneck:", x.shape)
        #print("Before bottleneck:", x.shape)
        # 使用瓶颈层处理特征
        
        ###768 12    384 24   192 48   96 96

        # 处理跨越连接
        # 对第二个跨越连接首先进行1*1卷积使通道数减少为原来的二分之一
        skip_connection_1 = self.reduce_channel_conv(skip_connections[1])
        # 对第二个跨越连接进行双线性插值上采样使得特征图尺寸扩大两倍
        skip_connection_1 = F.interpolate(skip_connection_1, scale_factor=2, mode='bilinear', align_corners=False)
        # 让第一个跨越连接与第二个处理后的跨越连接进行逐元素相乘，得到新的第一个跨越连接
        new_skip_connection_0 = skip_connections[0] + skip_connection_1   ##加号变成乘号就可以
        # 使用新的第一个跨越连接替换原始跨越连接列表中的第一个跨越连接
        #skip_connections[0] = new_skip_connection_0

        #处理第三个跨越连接的特征
        skip_connection_3 = self.dwconv1(skip_connections[2])
        skip_connection_3 = self.expand_skip_connection(skip_connection_3)
        new_skip_connection_4 = skip_connections[3] + skip_connection_3
        #skip_connections[3] = new_skip_connection_4


       #处理第二个跨越连接的特征
        skip_connection_21 = self.dwconv2(skip_connections[0])
        skip_connection_21 = self.expand_skip_connection2(skip_connection_21)   #将第一个跨层连接降采样
        new_skip_connection_21 = skip_connections[1] + skip_connection_21

        skip_connection_23 = self.reduce_channel_conv1(skip_connections[2]) 
        skip_connection_23 = F.interpolate(skip_connection_23, scale_factor=2, mode='bilinear', align_corners=False)
        new_skip_connection_23 = skip_connections[1] + skip_connection_23

        #skip_connections[1] = self.reduce_channel_conv1(skip_connections[1])

        #处理第三个跨越连接的数据
        skip_connection_32 = self.dwconv3(skip_connections[1])
        skip_connection_32 = self.expand_skip_connection3(skip_connection_32)   #将第一个跨层连接降采样
        new_skip_connection_32 = skip_connections[2] + skip_connection_32

        skip_connection_34 = self.reduce_channel_conv2(skip_connections[3]) 
        skip_connection_34 = F.interpolate(skip_connection_34, scale_factor=2, mode='bilinear', align_corners=False)
        new_skip_connection_34 = skip_connections[2] + skip_connection_34

        #skip_connections[2] = self.reduce_channel_conv2(skip_connections[2])
 
        z1 = skip_connections[0]
        z2 = skip_connections[1]
        z3 = skip_connections[2]
        z4 = skip_connections[3]
       



        ##### ""在这里啊面进行特征最后的融合""
        skip_connections[0] = torch.cat([new_skip_connection_0, skip_connections[0]], dim=1)# 384 12 
        skip_connections[3] = torch.cat([new_skip_connection_4, skip_connections[3]], dim=1)# 384 12 
        skip_connections[1] = torch.cat([new_skip_connection_21, new_skip_connection_23], dim=1)# 384 12 
        skip_connections[2] = torch.cat([new_skip_connection_32, new_skip_connection_34], dim=1)# 384 12 
       
        #加残差大小必须一致，所以必须加上这个内容，如果残差加在这里效果不好就算了
        ###""把z1的大小变大，算了还先变小试一下""
        # c1 = skip_connections[0]
        # c2 = skip_connections[1]
        # c3 = skip_connections[2]
        # c4 = skip_connections[3]

        # c1 = self.reduce_channel1(c1)
        # c2 = self.reduce_channel2(c2)
        # c3 = self.reduce_channel3(c3)
        # c4 = self.reduce_channel4(c4)


        # skip_connections[0] = z1 + c1
        # skip_connections[1] = z2 + c2
        # skip_connections[2] = z3 + c3
        # skip_connections[3] = z4 + c4


        # channel_attention1 = ChannelAttention(192).to('cuda:0')#.to('cpu')
        # channel_attention2 = ChannelAttention(384).to('cuda:0')#.to('cpu')
        # channel_attention3 = ChannelAttention(768).to('cuda:0')#.to('cpu')
        # channel_attention4 = ChannelAttention(1536).to('cuda:0')#.to('cpu')

        ### channel_skip_connection1 = self.channel_attention1(skip_connections[0])
        ### channel_skip_connection2 = self.channel_attention2(skip_connections[1])
        ### channel_skip_connection3 = self.channel_attention3(skip_connections[2])
        ### channel_skip_connection4 = self.channel_attention4(skip_connections[3])
        # skip_connections[0] = self.channel_attention1(skip_connections[0])
        # skip_connections[1] = self.channel_attention2(skip_connections[1])
        # skip_connections[2] = self.channel_attention3(skip_connections[2])
        # skip_connections[3] = self.channel_attention4(skip_connections[3])
        # print("channel_attention shape:",channel_skip_connection1.shape)
        # print("channel_attention shape:",channel_skip_connection2.shape)

        # spatial_attention = SpatialAttention().to('cuda')                     #.to('cuda:0')
        # spatial_skip_connection1 = spatial_attention(channel_skip_connection1)
        # spatial_skip_connection2 = spatial_attention(channel_skip_connection2)
        # spatial_skip_connection3 = spatial_attention(channel_skip_connection3)
        # spatial_skip_connection4 = spatial_attention(channel_skip_connection4)
        ### spatial_skip_connection1 = self.spatt1(channel_skip_connection1)
        ### spatial_skip_connection2 = self.spatt1(channel_skip_connection2)
        ### spatial_skip_connection3 = self.spatt1(channel_skip_connection3)
        ### spatial_skip_connection4 = self.spatt1(channel_skip_connection4)
        # print("spatial_attention shape:",spatial_skip_connection1.shape)
        # print("channel_attention shape:",spatial_skip_connection2.shape)
        # spatial_skip_connection2 = spatial_attention(channel_attention2)
        # spatial_skip_connection3 = spatial_attention(channel_attention3)
        # spatial_skip_connection4 = spatial_attention(channel_attention4)
        # spatial_skip_connection1 = spatial_attention(skip_connections[0])
        # spatial_skip_connection2 = spatial_attention(skip_connections[1])
        # spatial_skip_connection3 = spatial_attention(skip_connections[2])
        # spatial_skip_connection4 = spatial_attention(skip_connections[3])
        ### skip_connections[0] =  spatial_skip_connection1
        ### skip_connections[1] =  spatial_skip_connection2
        ### skip_connections[2] =  spatial_skip_connection3
        ### skip_connections[3] =  spatial_skip_connection4

        # skip_connections[0] =  (channel_skip_connection1 + spatial_skip_connection1)
        # skip_connections[1] =  (channel_skip_connection2 + spatial_skip_connection2)
        # skip_connections[2] =  (channel_skip_connection3 + spatial_skip_connection3)
        # skip_connections[3] =  (channel_skip_connection4 + spatial_skip_connection4)
        # print("skip_connections[0] shape:",skip_connections[0].shape)

        ###""通道数在上面就变好了""
        skip_connections[0] = self.reduce_channel1(skip_connections[0])
        skip_connections[1] = self.reduce_channel2(skip_connections[1])
        skip_connections[2] = self.reduce_channel3(skip_connections[2])
        skip_connections[3] = self.reduce_channel4(skip_connections[3])

        ###""去掉这几个部分，最原始的作用,relu本来就是去掉的、要是不加这个特征融合会不会的结果会更好呢？""
        # skip_connections[0] = self.relu(skip_connections[0])
        # skip_connections[1] = self.relu(skip_connections[1])
        # skip_connections[2] = self.relu(skip_connections[2])

        # skip_connections[3] = self.relu(skip_connections[3])
        
        skip_connections[0] = z1 + skip_connections[0]
        skip_connections[1] = z2 + skip_connections[1]
        skip_connections[2] = z3 + skip_connections[2]
        skip_connections[3] = z4 + skip_connections[3]




        x = self.down_lay(x) #1536 6


        x = self.bottleneck(x)  #1536 6
        
        #print("After bottleneck:", x.shape)
        # 解码器部分
        #for i in range(3):
        #    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #    x = torch.cat([x, skip_connections[2 - i]], dim=1)
        #    x = self.decoder[i](x)

        ###resc1 = self.res[0](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        x = self.decoder[0](x)  #768 12

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn1(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc1

        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 
        # x = self.block1(x)

        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[3] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)


        ###resc2 = self.res[1](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24

        x = self.decoder[1](x)  #384 24

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn2(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc2

        
        #print("Before bottleneck:", x.shape)
        
        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24
        # x = self.block2(x)
        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[2] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        ###resc3 = self.res[2](x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        x = self.decoder[2](x) #192 48

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn3(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc3
        
        
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        # x = self.block3(x)

        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[1] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        ###resc4 = self.res[3](x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        x = self.decoder[3](x)  # 96 96
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn4(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        ###x= x + resc4
        
        
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        # x = self.block4(x)

        x = x.flatten(2).transpose(1, 2) #解释一下这里的操作 z1大小是8,96,128,128，将其变成8,128,96,128 
        x = self.swint_layers[0] (x)
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0,3,1, 2)

        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn5(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)


        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
        #x = torch.cat([x, skip_connections[0][:, :, :x.shape[2], :x.shape[3]]], dim=1)
        #x = self.decoder[3](x)
        #print("After bottleneck:", x.shape)

        x = self.final_conv(x)  #4 384 384 
        #print("Before bottleneck:", x.shape)
        return x
    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            #model_dict = self.encoder.state_dict()
            model_dict = self.state_dict()


            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict = {k.replace("head.", ""): v for k, v in pretrained_dict.items() if k.startswith("head.") or k in model_dict}
            #model_dict.update(pretrained_dict)
            #model.load_state_dict(pretrained_dict,)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            #msg = model.load_state_dict(pretrained_dict, strict=False)
            #print("missing keys:", msg.missing_keys)

        else:
            print("No pretrained model provided, training from scratch.")
    """
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.state_dict()

            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict_e = {}
            for k, v in pretrained_dict['model'].items():
                if 'stages' in k:
                    pretrained_dict_e[k.replace('stages', 'encoder.stages')] = v
                elif 'downsample' in k:
                    pretrained_dict_e[k.replace('downsample_layers', 'encoder.downsample_layers')] = v
                elif 'norm' in k:
                    pretrained_dict_e[k.replace('norm', 'encoder.norm')] = v

            for key, value in pretrained_dict_e.items():
                if key in model_dict.keys():
                    model_dict[key] = value

            self.load_state_dict(model_dict, strict=False)
        else:
            print("No pretrained model provided, training from scratch.")





"""
class UNetConvNeXtV2Base(nn.Module):
    def __init__(self, in_chans=3, num_classes=4):
        super().__init__()
        self.encoder = convnextv2_base(in_chans=in_chans, num_classes=None,include_head=False)

        # 使用 ConvNeXtV2 Tiny 的第一个阶段作为瓶颈层
        #self.bottleneck = self.encoder.stages[0]
        # 修改bottleneck层为深度为3的ConvNeXtV2
        self.bottleneck = nn.Sequential(
            *[Block(dim=2048, drop_path=0.) for _ in range(3)]
        )

        self.down_lay = nn.Sequential(
                LayerNorm(1024, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(1024, 2048, kernel_size=2, stride=2),
        )

        # 解码器部分
        #####"#"#"#########
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])
        #"##"
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])



        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        #print("Before bottleneck:", x.shape)
        # 编码器部分
        for i in range(4):
            x = self.encoder.downsample_layers[i](x)   ###下面两个换一下顺序
            
            x = self.encoder.stages[i](x)
            skip_connections.append(x)
            #print("Before bottleneck:", x.shape)
        #print("Before bottleneck:", x.shape)
        # 使用瓶颈层处理特征
        
        ###768 12    384 24   192 48   96 96

        x = self.down_lay(x) #1536 6


        x = self.bottleneck(x)  #1536 6
        
        #print("After bottleneck:", x.shape)
        # 解码器部分
        #for i in range(3):
        #    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #    x = torch.cat([x, skip_connections[2 - i]], dim=1)
        #    x = self.decoder[i](x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        x = self.decoder[0](x)  #768 12
        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        x = self.decoder[1](x)  #384 24

        
        #print("Before bottleneck:", x.shape)
        
        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        x = self.decoder[2](x) #192 48
        
        
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        x = self.decoder[3](x)  # 96 96
        
        
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)


        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
        #x = torch.cat([x, skip_connections[0][:, :, :x.shape[2], :x.shape[3]]], dim=1)
        #x = self.decoder[3](x)
        #print("After bottleneck:", x.shape)

        x = self.final_conv(x)  #4 384 384 
        #print("Before bottleneck:", x.shape)
        return x
    "#""
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            #model_dict = self.encoder.state_dict()
            model_dict = self.state_dict()


            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict = {k.replace("head.", ""): v for k, v in pretrained_dict.items() if k.startswith("head.") or k in model_dict}
            #model_dict.update(pretrained_dict)
            #model.load_state_dict(pretrained_dict,)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            #msg = model.load_state_dict(pretrained_dict, strict=False)
            #print("missing keys:", msg.missing_keys)

        else:
            print("No pretrained model provided, training from scratch.")
    ""#"
    def load_from(self, pretrained_model_path):
        if pretrained_model_path is not None:
            print(f"Loading pretrained weights from {pretrained_model_path}")
            pretrained_dict = torch.load(pretrained_model_path, map_location='cpu')
            model_dict = self.state_dict()

            # Filter out unnecessary keys and rename keys to match encoder state dict
            pretrained_dict_e = {}
            for k, v in pretrained_dict['model'].items():
                if 'stages' in k:
                    pretrained_dict_e[k.replace('stages', 'encoder.stages')] = v
                elif 'downsample' in k:
                    pretrained_dict_e[k.replace('downsample_layers', 'encoder.downsample_layers')] = v
                elif 'norm' in k:
                    pretrained_dict_e[k.replace('norm', 'encoder.norm')] = v

            for key, value in pretrained_dict_e.items():
                if key in model_dict.keys():
                    model_dict[key] = value

            self.load_state_dict(model_dict, strict=False)
        else:
            print("No pretrained model provided, training from scratch.")
"""
