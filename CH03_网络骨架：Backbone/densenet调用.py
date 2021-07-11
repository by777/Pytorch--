# -*- coding: utf-8 -*-
# @TIME : 2021/7/11 16:55
# @AUTHOR : Xu Bai
# @FILE : densenet调用.py
# @DESCRIPTION :
import torch

from densenet_block import Denseblock

# 包含6个bottleneck
denseblock = Denseblock(64, 32, 6)
print(denseblock)
print('第1个输bottleneck的输入通道为64，输出固定为32')
print('第2个输bottleneck的输入通道为96，输出固定为32')
print('第3个输bottleneck的输入通道为128，输出固定为32')
print('第4个输bottleneck的输入通道为160，输出固定为32')
print('第5个输bottleneck的输入通道为192，输出固定为32')
print('第6个输bottleneck的输入通道为224，输出固定为32')
input = torch.randn(1, 64, 256, 256)
output = denseblock(input)
print('输出通道数为：224 + 32 = 64 + 32 x 6 = 256')
print(output.shape)
