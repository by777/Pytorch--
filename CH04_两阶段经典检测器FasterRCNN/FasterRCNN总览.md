# Faster RCNN

主要分为：

+ 特征提取网络
+ RPN模块
+ RoI Pooling
+ RCNN模块

# 特征提取网络Backbone

输入图像首先经过Backbone得到特征图，在此以VGGNet为例，假设输入的图像为3 x 600 x 800，由于VGGNet下采样率为16，因此输出的feature map的维度为512 x 37 x 50。

# RPN模块

**区域生成模块。其作用是生成较好的建议框，即Proposal**，这里用到了强先验的Anchor。RPN包括了5个子模块：

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

Anchor的本质是在原图大小上的一系列的矩形框，但Faster RCNN将这一系列的矩形框与feature map进行了关联。具体做法是，首先对feature map进行3 x 3卷积，得到每一个点的维度是512维，这512维的数据对应原始图像上的很多不同的大小与宽高区域的特征，这些区域的中心点都相同。如果下采样率维默认的16，则每一个点的而坐标乘以16即可得到对应的原始坐标。

为了适应不同物体的宽高，在作者的论文中，默认每一个点上抽取了9种Anchors，具体Scale为{8，16，32}，Ratio为{0.5，1，2}，将9种Anchors的大小反算到原图上，即得到不同原始Proposal。由于feature map的大小为37 x 50，因此一共有37 x 50 x 9 = 16650个Anchors。而后通过分类网络与回归网络得到每一个Anchor的前景背景概率和偏移量，前景背景概率原来判断Anchor是前景的概率，回归网络则将预测偏移量作用到Anchor使得Anchor更接近真实物体坐标。

![image-20210719103226550](https://raw.githubusercontent.com/by777/imgRep/main/img/20210719103226.png)

```python
def generate_anchors(base_size=16,ratios=[0.5,1,2],scales=2**np.arange(3,6)):
    # 首先创建一个基本anchor为[0,0,15,15]
    base_anchor = np.arange([1,1,base_size,base_size]) - 1
    # 将基本anchor进行宽高变换，生成3种宽高比的s:Anchor
    ratio_anchor = _ratio_enum(base_anchor,ratio)
    # 将上述anchor尺度变换，得到最终9种Anchors
    anchors = np.vstack([_scale_enum(ratio_anchors[i,:],scales) for i in xrange(ratio_anchors.shape[0])])
    # 返回对应于feature map大小的anchors
    return anchors
```

# RPN的真值与预测值

对于物体检测任务来说，模型需要预测物体的类别与出现的位置。即类别、中心点坐标x和y、w和h，5个量。由于有了anchor先验框，RPN可以预测Anchor的类别作为预测边框的类别，并且可以预测真实的边框相对于anchor的偏移量，而不是直接预测边框的中心点坐标和宽高（x,y,w,h）。

举个例子，输入图像中有3个Anchors与两个标签，从位置来看，Anchor A，C分别于标签M、N有一定的重叠，而Anchor B的位置更像是背景。

![image-20210720095156308](https://raw.githubusercontent.com/by777/imgRep/main/img/20210720095156.png)

+ 首先介绍**模型的真值**。

  对于类别的真值，由于RPN只负责区域生成，保证Recall，而没必要细分每一个区域属于哪一个类别，因此只需要前景与背景两个类别，前景即有物体，背景则没有物体。

  RPN通过计算Anchor与标签的IoU来判断Anchor是属于前景还是背景，**当IoU大于一定值时，该Anchor的真值为前景**，低于一定值时，该Anchor的真值为背景。

+ 然后是**偏移量的真值**。

  仍以上图的Anchor A与label M为例，假设Anchor A中心坐标为X_a与Y_a，宽和高为W_a和H_a，label M的中心坐标为x和y，宽高为w和h，则对应的偏移真值计算公式如下：
  $$
  t_x = (x - x_a) / w_a \\
  t_y = (y - y_a) / h_a \\
  t_w = log(\frac{w}{w_a}) \\
  t_h = log(\frac{h}{h_a}) \\
  $$
  从上式中可以看出，**位置偏移t_x与t_y利用宽高进行了归一化，而宽高偏移t_w和t_h进行了对数处理**，这样做的好处在于进一步限制了偏移量的范围，便于预测。

有了上述的真值，为了求取损失，RPN通过CNN分别得到了类别与偏移量的预测值。具体来讲，RPN需要预测每一个Anchor属于前景后景概率，同时还需要预测真实物体相对于Anchor的偏移量，记为
$$
t_x^*、t_y^*、t_w^*、t_h^*
$$
另外，在得到预测偏移量后，可以使用下面的公式讲预测偏移量作用到对应的Anchor上，得到预测框的实际位置
$$
x^*、y^*、w^*、h^*
$$

$$ \begin{aligned}
t_x^* = (x - x_a) / w_a  
t_y^* = (y - y_a) / h_a  
t_w^* = log(\frac{w}{w_a})  
t_h^* = log(\frac{h}{h_a}) 
\end{aligned}
$$ 

