# -*- coding: utf-8 -*-
# @TIME : 2021/7/6 20:55
# @AUTHOR : Xu Bai
# @FILE : vgg_调用.py
# @DESCRIPTION :
import torch

from vgg import VGG

vgg = VGG(21)
input = torch.randn(1, 3, 224, 224)
print(input.shape)
scores = vgg(input)
print(scores)
print(scores.shape)
print(vgg.features)