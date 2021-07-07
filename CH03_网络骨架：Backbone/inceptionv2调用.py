# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 16:07
# @AUTHOR : Xu Bai
# @FILE : inceptionv2调用.py
# @DESCRIPTION :
import torch

from inceptionv2 import Inceptionv2

net = Inceptionv2()
input = torch.randn(2, 192, 32, 32)
out = net(input)
print(out)
