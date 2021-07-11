# DenseNet

ResNet缓解了梯度消失。DenseNet将前后层Shortcuts前后层交流，通过建立前面所有层与后面层的密集连接，实现了通道维度上的复用。使其**可以在参数参与计算量更小的情况下实现比ResNet更优的性能。**

DenseNet的网络架构，由多个Dense Block与中间的卷积层池化组成。核心就在Dense Block中。DenseBlock中的黑点代表一个卷积层，其中的多条黑线代表数据的流动，每一层的输入都由前面所有的卷积层输出组成。注意这里使用了通道拼接Concatate操作，而非ResNet逐元素相加。

# DenseNet的特性

+ 神经网络一般需要池化等操作缩小特征图尺寸来提取语义特征，而Dense Block需要保持每一个Block内的特征图尺寸一致来进行Concatenate操作，因此DenseNet被分成了多个Block，一般4个。
+ 两个相邻的Dense Block之间的部分被称为Transition层，具体包括BN，ReLU，1x1卷积，2x2平均池化。1x1卷积为了降维，起到压缩模型的作用，而平均池化则是降低特征图尺寸。

![image-20210711163704355](https://raw.githubusercontent.com/by777/imgRep/main/img/20210711163704.png)

# 关于Block，有4个特性需要注意

+ 每一个Bottleneck输出的特征通道数是一致的，例如上面的32，。同时可以看到，通过concatenate操作后通道数是按32的增长量来计算的，因此32也称为GrowthRate。
+ 这里1x1卷积的作用是固定输出通道数，达到降维的作用。当几十个bottlenek相连接时，Concatnate后的通道数会增加到上千，如果不增加1x1卷积来降维，后续3x3卷积所需的参数量会急剧增加。1x1卷积的通道数通常是GrowthRate的4倍。
+ Block采用了激活函数在前、卷积层在后的顺序，这与一般的网络是不同的。

# 优势
+ 密集连接的网络，使得每一层都会接受到其后所有层的梯度，而不是像普通卷积链式的反传，因此一定程度上解决了梯度消失的问题
+ 通过Concatenate使得大量特征被复用，每个层独有的特征图的通道数是少的，因此相比Resnet，它的参数量更少

# 不足
多次的concatenate使得数据被复制多次，显存增加的很快，需要一定的显存优化技术。另外Resnet应用更广泛一些。
