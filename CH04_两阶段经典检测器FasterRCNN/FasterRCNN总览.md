# Faster RCNN

主要分为：

+ 特征提取网络
+ RPN模块
+ RoI Pooling
+ RCNN模块

## 特征提取网络Backbone

输入图像首先经过Backbone得到特征图，在此以VGGNet为例，假设输入的图像为3 x 600 x 800，由于VGGNet下采样率为16，因此输出的feature map的维度为512 x 37 x 50。

## RPN模块

区域生成模块。其作用是生成较好的建议框，即Proposal，这里用到了强先验的Anchor。RPN包括了5个子模块：

+ Anchor生成。RPN对feature map上的每一个点都对应了9个Anchor，这9个Anchor的大小宽高不同，对应到原图基本可以覆盖所有可能出现的物体。因此，有了数量庞大的Anchor，RPN的接下来任务就是从中筛选，并调整更好的位置，得到Proposal。
+ RPN网络。