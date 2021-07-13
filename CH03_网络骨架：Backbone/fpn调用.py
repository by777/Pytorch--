# -*- coding: utf-8 -*-
# @TIME : 2021/7/12 20:41
# @AUTHOR : Xu Bai
# @FILE : fpn调用.py
# @DESCRIPTION :
from fpn import FPN
import torch

net_fpn = FPN([3, 4, 6, 3])

print('FPN的第一个卷积层：')
print(net_fpn.conv1)

print('FPN的第一个BN层：')
print(net_fpn.bn1)

print('FPN的第一个relu层：')
print(net_fpn.relu)

print('FPN的第一个池化层：')
print(net_fpn.maxpool)

print('\n')
print('FPN的第一个layer,即前面的C2，包含了3个Bottleneck')
print(net_fpn.layer1)

print('\n')
print('FPN的第二个layer,即前面的C3，包含了4个Bottleneck')
print(net_fpn.layer2)

print('\n')
print('FPN的第三个layer,即前面的C4，包含了6个Bottleneck')
print(net_fpn.layer3)

print('\n')
print('1x1卷积，用以得到P5')
print(net_fpn.toplayer)

print('\n')
print('对P4平滑处理的卷积层')
print(net_fpn.smooth1)

print('\n')
print('对C4横向处理的卷积层')
print(net_fpn.latlayer1)

input = torch.randn(1, 3, 224, 224)
output = net_fpn(input)
print('\n')
print('返回的P2、P3、P4、P5，这四个特征图通道数相同，但是特征图尺寸递减')
print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)
