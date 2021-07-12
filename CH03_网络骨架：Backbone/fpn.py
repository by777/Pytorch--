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


# FPN的类，初始化需要一个list，代表ResNet每一个阶段的Bottleneck的数量
class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 64

        # 处理输入的C1模块
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # 搭建自下而上的C2、C3、C4、C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)

        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 10)

        # 3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)

        # 横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 516, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 516, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 516, 1, 1, 0)

    # 构建C2到C5，注意区分stride值为1和2的情况
    def _make_layer(self, planes, blocks, stride=1):
        pass


if __name__ == '__main__':
    neck = Bottleneck(3, 4, 6, 3)
    print(neck)
