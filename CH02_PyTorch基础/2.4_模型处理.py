# -*- coding: utf-8 -*-
# @TIME : 2021/7/7 10:50
# @AUTHOR : Xu Bai
# @FILE : 2.4_模型处理.py
# @DESCRIPTION :
from torch import nn
from torchvision import models

vgg = models.vgg16()
print(len(vgg.features),len(vgg.classifier))

print(vgg.classifier[-1])