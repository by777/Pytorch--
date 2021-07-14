# -*- coding: utf-8 -*-
# @TIME : 2021/7/14 16:30
# @AUTHOR : Xu Bai
# @FILE : detnet调用.py
# @DESCRIPTION :
import torch
from torch import nn
from detnet_bottkeneck import DetBottleneck

# 完成一个stage 5，即B-A-A结构，Stage 4输出通道数为1024
bottleneck_b = DetBottleneck(1024, 256, 1, True)
print(bottleneck_b)

bottleneck_a1 = DetBottleneck(256, 256)
bottleneck_a2 = DetBottleneck(256, 256)

input = torch.randn(1, 1024, 14, 14)

output1 = bottleneck_b(input)
output2 = bottleneck_a1(output1)
output3 = bottleneck_a2(output2)

print(output1.shape, output2.shape, output3.shape)
