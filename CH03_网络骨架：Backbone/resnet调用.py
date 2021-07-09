# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 15:54
# @AUTHOR : Xu Bai
# @FILE : resnet调用.py
# @DESCRIPTION :
import torch

from resnet_bottleneck import Bottleneck

bottleneck_1_1 = Bottleneck(64, 256)
print(bottleneck_1_1)
input = torch.randn(1, 64, 56, 56)
output = bottleneck_1_1(input)
print(input.shape)
print(output.shape)
print('相比输入，输出的特征图的分辨率没变，而通道数变为4倍')
