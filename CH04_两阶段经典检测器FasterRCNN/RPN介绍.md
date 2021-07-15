# RPN

+ 输入：feature map、物体标签，即训练集中所有物体的类别和标签位置
+ 输出：Proposal、分类Loss、回归Loss、其中，Proposal作为生成的区域，供后续模块分类和回归。两部分损失用作优化网络。

```python
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
    ...S
```

