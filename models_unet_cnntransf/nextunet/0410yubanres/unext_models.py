# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN
#import contextlib

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
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
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
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


class UNetConvNeXtV2Tiny(nn.Module):
    def __init__(self, in_chans=3, num_classes=4):
        super().__init__()
        self.encoder = convnextv2_tiny(in_chans=in_chans, num_classes=None,include_head=False)

        # 使用 ConvNeXtV2 Tiny 的第一个阶段作为瓶颈层
        #self.bottleneck = self.encoder.stages[0]
        # 修改bottleneck层为深度为3的ConvNeXtV2
        self.bottleneck = nn.Sequential(
            *[Block(dim=1536, drop_path=0.) for _ in range(3)]
        )

        self.down_lay = nn.Sequential(
                LayerNorm(768, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(768, 1536, kernel_size=2, stride=2),
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
                nn.Conv2d(1536, 768, kernel_size=3, padding=1),
                #nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Conv2d(768, 768, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map size
                #nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1536, 384, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(384, 384, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
            ),
            nn.Sequential(
                nn.Conv2d(768, 192, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map size
            ),
            nn.Sequential(
                nn.Conv2d(384, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map size
            ),
            nn.Sequential(
                nn.Conv2d(192, 96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(96,96, kernel_size=3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(96, 96, kernel_size=4, stride=4)  # Add a transpose convolution layer to double the feature map size
            ),
            #####是否可以换成gelu
            #nn.Sequential(
            #    nn.Conv2d(192, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True),
            #    nn.Conv2d(96, 96, kernel_size=3, padding=1),
            #    nn.ReLU(inplace=True)
            #)
        ])



        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)

        self.grn1 = GRN(768)
        self.grn2 = GRN(384)
        self.grn3 = GRN(192)
        self.grn4 = GRN(96)
        self.grn5 = GRN(96)


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

        x = self.down_lay(x) #1536 6  下采样


        x = self.bottleneck(x)  #1536 6
        
        #print("After bottleneck:", x.shape)
        # 解码器部分
        #for i in range(3):
        #    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #    x = torch.cat([x, skip_connections[2 - i]], dim=1)
        #    x = self.decoder[i](x)
        #####x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        x = self.decoder[0](x)  #768 12
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn1(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 

        #####x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        x = self.decoder[1](x)  #384 24
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn2(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        
        #print("Before bottleneck:", x.shape)
        
        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24
        #####x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        x = self.decoder[2](x) #192 48
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn3(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        #####x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        x = self.decoder[3](x)  # 96 96
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn4(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn5(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        #####x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
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

        self.res = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1),
                nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1),
                nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, stride=1),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Add a transpose convolution layer to double the feature map siz
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


        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

        self.grn1 = GRN(1024)
        self.grn2 = GRN(512)
        self.grn3 = GRN(256)
        self.grn4 = GRN(128)
        self.grn5 = GRN(128)


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
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 12
        resc1 = self.res[0](x)
        x = self.decoder[0](x)  #768 12
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn1(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x= x + resc1

        x = torch.cat([x, skip_connections[3]], dim=1)#1536 12 

        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #1536 24
        resc2 = self.res[1](x)
        x = self.decoder[1](x)  #384 24

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn2(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x= x + resc2

        
        #print("Before bottleneck:", x.shape)
        
        x = torch.cat([x, skip_connections[2]], dim=1)  #768 24
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  #768 48
        resc3 = self.res[2](x)
        x = self.decoder[2](x) #192 48

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn3(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x= x + resc3
        
        
        x = torch.cat([x, skip_connections[1]], dim=1) #384 48
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) #384 96
        resc4 = self.res[3](x)
        x = self.decoder[3](x)  # 96 96
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn4(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x= x + resc4
        
        
        x = torch.cat([x, skip_connections[0]], dim=1)  #192 96
        x = self.decoder[4](x) #96 96
        #print("After bottleneck:", x.shape)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.grn5(x)  ###########加入全局响应归一化层
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)


        #x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) #96 384
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
