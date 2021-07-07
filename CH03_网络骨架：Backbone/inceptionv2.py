# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 15:55
# @AUTHOR : Xu Bai
# @FILE : inceptionv2.py
# @DESCRIPTION :
import torch
from torch import nn
from torch.nn import functional as F


# 构建的基础卷积模块，增加了BN层

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inceptionv2(nn.Module):
    def __init__(self):
        super(Inceptionv2, self).__init__()
        # 1*1卷积
        self.branch1 = BasicConv2d(192, 96, 1, 0)
        # 1 * 1 和 3*3卷积分支
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 48, 1, 0),
            BasicConv2d(48, 64, 3, 1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(192, 64, 1, 0),
            BasicConv2d(64, 96, 3, 1),
            BasicConv2d(96, 96, 3, 1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, 1, 0)
        )

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), dim=1)
        return out
