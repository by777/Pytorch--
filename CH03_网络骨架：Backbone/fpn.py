# -*- coding: utf-8 -*-
# @TIME : 2021/7/12 11:14
# @AUTHOR : Xu Bai
# @FILE : fpn.py
# @DESCRIPTION :
from torch import nn


# Resnet的基本Bottleneck类
class Bottleneck(nn.Module):
    expansion = 4  # 通道倍增数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


if __name__ == '__main__':
    neck = Bottleneck(3,4,6,3)
    print(neck)