# -*- coding: utf-8 -*-
# @TIME : 2021/7/13 21:41
# @AUTHOR : Xu Bai
# @FILE : detnet_bottkeneck.py
# @DESCRIPTION :
import torch
from torch import nn


class DetBottleneck(nn.Module):
    # 初始化时extra为False时为Bottleneck A, 为B时为Bottleneck B
    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        # 构建连续3个卷积层的Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        # Bottleneck B为1x1卷积
        if self.extra:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out
