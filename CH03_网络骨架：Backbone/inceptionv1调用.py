# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 15:46
# @AUTHOR : Xu Bai
# @FILE : inceptionv1调用.py
# @DESCRIPTION :
import torch

from inceptionv1 import Inceptionv1

net_inceptionv1 = Inceptionv1(3, 64, 32, 64, 64, 96, 32)
print(net_inceptionv1)

input = torch.rand(2,3,256,256)
# print(input)
output = net_inceptionv1(input)
print(output.shape)