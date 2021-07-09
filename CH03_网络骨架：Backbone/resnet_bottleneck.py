# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 17:08
# @AUTHOR : Xu Bai
# @FILE : resnet_bottleneck.py
# @DESCRIPTION :
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()

        # 网络堆叠层是由  1x1 3x3 1x1 这3个卷积组成的，中间包含BN层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.relu = nn.ReLU(inplace=True)
        # 利用下采样结构把恒等映射的通道数映射为与卷积堆叠层相同，保证可以相加
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        identity = self.downsample(x)
        ########################################################
        # 将identity(恒等映射)与网络堆叠层输出相加，并经过ReLU后输出 #
        ########################################################
        out = out + identity
        out = self.relu(out)
        return out
