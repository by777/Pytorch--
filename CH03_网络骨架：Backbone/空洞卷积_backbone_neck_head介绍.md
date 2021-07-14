[参考连接]: https://www.zhihu.com/zvideo/1356732245315805184



# 空洞卷积（Dilated Convolution）

空洞卷积最初是为解决图像分割问题而提出的。常见的图像分割算法通常使用池化层来增大感受野。同时也缩小了特征图尺寸，然后利用上采样还原图像尺寸。特征图缩小再放大造成了精度上的损失，因此需要一种操作能够在增加感受野的同时特征图的尺寸不变，从而代替池化和上采样操作。因此，空洞卷积就诞生了。

**空洞卷积**，卷积核中带有一些洞，跳过一些元素进行卷积。在代码实现时，空洞卷积有一个额外的超参数dilation rate默认为1，表示空洞数。

# Backbone、neck和head

在One-Stage Anchor free的检测器中，我们习惯性的将整个网络划分为3个部分：`backbone` 、`neck`和`head`。

## backbone: VGG\ResNet\ResNeXt\EfficientNet

当前物体检测算法各不相同，但第一步通常是利用CNN处理图像，生成特征图，然后再利用各种算法完成区域生成和损失计算，这部分CNN是整个检测算法的“骨架”，也称为Backbone。

每一个被选择特征图都有内在固有的语义表达能力。即：使用这个特征图做后面的预测，它到底能学什么、能学多好，可能就已经内定了。

特征图上每个位置上的感受野已经确定了，也就是它看到了什么区域已经非常明确了，让它去预测超过这个区域的目标，其实就不合理了。看到了但是能不能学好又是另一回事了。这个就和backbone的结构设计就有很大关系了。

**总结：backbone能为检测提供若干种感受野大小和中心步长的组合，以满足对不同尺度和类别的目标检测。**

## neck：Naiveneck\FPN\BiFPN\PANet\NAS-FPN

neck接受来自backbone的若干个特征图，处理后再输出给head。NaiveNeck：其实也就是没有neck，如SSD。

neck的第一要务就是进行特征融合：具有不同感受野大小的特征图进行了耦合，从而增强了特征图的表达能力。neck决定了head的数量，不同尺度的目标被分配到不同的Head学习，即：学习的负担被分散到了多个层级的特征图上。此外，进行宽带对齐，便于后续使用。

## head：RetinaNet-Head\FCOS-Head

划分方式1：有无Anchor：RetinaNet就是anchor-based，同时没有quality分支

划分方式2：有无quality分支：FCOS：anchor-free的，同时有quality分支

head通常会分为分类分支和回归分支，且两个分支独立不共享权重（但不一定）。

![image-20210714170257516](https://raw.githubusercontent.com/by777/imgRep/main/img/20210714170257.png)





