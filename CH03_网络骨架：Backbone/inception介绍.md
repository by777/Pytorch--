# Inception v1

一般来说，增加网络的深度与宽度可以增加网络的性能。但是这样也会增加网络的参数量，同时较深的网络需要较多的数据，否则容易出现过拟合。除此之外，还有可能梯度消失。

Inception v1是一个精心设计的22层网络结构，并提出了具有良好局部特征结构的Inception模块，即对特征并行的执行多个大小不同的卷积运算和池化。最后**拼接**到一起。

由于1 x 1、3 x 3、5 x 5的卷积运算对应不同的特征图区域，因此可以得到更好的图像表征信息。

![image-20210707203212687](https://raw.githubusercontent.com/by777/imgRep/main/img/20210707203213.png)

Inception模块如图所示，使用了3个大小不同的卷积核进行卷积运算，同时还有一个最大池化。然后将这4部分级联起来（通道拼接）送入下一层。

在上述基础上，为了进一步降低网络藏书量，Inception增加了多个 1 x 1卷积。这种1 x 1卷积块可以先将特征图降维，再送入 3 x 3和5 x 5的卷积核，由于通道数降低，参数量也显著减少。

![image-20210707203949673](https://raw.githubusercontent.com/by777/imgRep/main/img/20210707203949.png)

<u>Inceptionv1是AlexNet的1/12，VGG的1/3，适合处理大规模数据，尤其是计算资源有限的平台。</u> 

# Inception v2

Inception v2通过卷积分解与正则化实现更高效的计算，增加了BN层，同时利用两个级联的3 x 3卷积取代了Inception v1的 5 x 5卷积。这种方式即减少了卷积参数量，又增加了网络的非线性能力。

![image-20210707204432809](https://raw.githubusercontent.com/by777/imgRep/main/img/20210707204432.png)

# Inception v3

Inception v3在v2的基础上使用了RMSProp优化器，在辅助的分类器部分增加了 7 x 7的卷积，并且使用了标签平滑技术。

# Inception v4

Inception v4将inception的思想与残差网络结合。
