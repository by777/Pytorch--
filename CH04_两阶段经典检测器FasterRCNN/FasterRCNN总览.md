# Faster RCNN

主要分为：

+ 特征提取网络
+ RPN模块
+ RoI Pooling
+ RCNN模块

# 特征提取网络Backbone

输入图像首先经过Backbone得到特征图，在此以VGGNet为例，假设输入的图像为3 x 600 x 800，由于VGGNet下采样率为16，因此输出的feature map的维度为512 x 37 x 50。

# RPN模块

区域生成模块。其作用是生成较好的建议框，即Proposal，这里用到了强先验的Anchor。RPN包括了5个子模块：

1. **Anchor生成**。RPN对**feature map上的每一个点都对应了9个Anchor**，这9个Anchor的大小宽高不同，对应到原图基本可以覆盖所有可能出现的物体。因此，有了数量庞大的Anchor，RPN的接下来任务就是从中筛选，并调整更好的位置，得到Proposal。
2. **RPN网络**。与上面的Anchor对应，由于feature map上每个点都对应了9个anchor，因此可以利用1 x 1卷积再feature map上**得到每一个Anchor的预测得分与预测偏移值**。
3. **计算RPN Loss**。这一步旨在训练中，将所有的Anchors与标签匹配，匹配程度较好的赋予正样本，较差的赋予负样本，得到分类与偏移的真值，与第二步中的预测得分与预测偏移值进行loss的计算。
4. **生成Proposal**。利用第二步中每一个Anchor的预测得分与偏移量，可以进一步得到一组较好的Proposal，送到后续网络中。
5. **筛选Proposal得到RoI**。在训练时，由于Proposal的数量还是太多（默认是2000），需要进一步筛选得到RoI（默认256个）。在测试阶段，则不需要该模块，Proposal可以直接作为RoI，默认数量是300。
6. **RoI模块**。这部分承上启下，接受CNN提取的feature map和RPN的RoI。输出送到RCNN网络中。由于RCNN模块采用了全连接网络，要求特征的维度固定，而每一个RoI对应的特征大小各不相同，无法送入全连接网络，因此**RoI Pooling将RoI的特征池化到固定的维度，方便送到全连接网络中**。
7. **RCNN模块**。将RoI得到的特征送入全连接网络，预测每一个RoI的分类，并预测偏移量以精修边框位置，并计算损失，完成整个Faster RCNN过程。主要包含3个部分：
   - **RCNN全连接网络**。将得到的固定维度的RoI特征接到全连接网络中，输出为RCNN部分的预测得分与预测回归偏移量。
   - **计算RCNN的真值**。对应筛选出的RoI，需要确定是正样本还是负样本，同时 计算真实物体的偏移量。在实际实现时，为实现方便，这一步往往与RPN最后实现RoI那一步放到一起。
   - **RCNN Loss**。通过RCNN的预测值与RoI部分的真值，计算分类与回归Loss。

通过整个过程可以看出，Faster RCNN是一个两阶段的算法，即RPN与RCNN，这两步都需要计算损失，只不过前者还要为后者提供较好的感兴趣区域。

```python
# RPN
# 输入：feature map、物体标签，即训练集中所有物体的类别和标签位置
# 输出：Proposal、分类Loss、回归Loss、其中，Proposal作为生成的区域，供后续模块分类# 和回归。两部分损失用作优化网络。
def forward(self, im_data, im_info, gt_boxes, num_boxes):
    # 输入数据的第一维是batch数
    batch_size = im_data.size()
    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    # 从VGGNet的backbone获取feature map
    base_feat = self.RCNN_base(im_data)
    # 将feature map送入RPN得到Proposal与分类与回归Loss
    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
    ...
```

# 理解Anchor

Anchor的本质是在原图大小上的一系列的矩形框，但Faster RCNN将这一系列的矩形框与feature map进行了关联。具体做法是，
