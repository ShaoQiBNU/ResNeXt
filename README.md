ResNeXt网络详解
===============

# 一. ResNeXt简介

> ResNeXt是一篇发表在2017CVPR上的论文，作者提出 ResNeXt 的主要原因在于：传统的要提高模型的准确率，都是加深或加宽网络，但是随着超参数数量的增加（比如channels数，filter size等等），网络设计的难度和计算开销也会增加。因此本文提出的 ResNeXt 结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量（得益于子模块的拓扑结构一样）。

# 二. 网络结构

> ResNeXt跟Inception-ResNet在结构上还是有点相似的，实际上就是将group convolution引进了ResNet中，以获得更少的参数。不过说起group convolution要回到2012的AlexNet了，那时候由于GPU并没有现在的强大，所以作者把网络拆成两部分，不过现在都慢慢淡出人们的视野了，然而KaiMing却造就了又一个经典–ResNeXt。 

## (一) cardinality

> VGG网络主要采用堆叠网络来提高模型精度；ResNet网络也借用了这样的思想，不过增加了short-cut，扩展了宽度；GoogleNet的Inception模型则采用了split-transform-merge策略——input分别做1x1 3x3 5x5卷积，然后通过concatenation做merge处理。ResNeXt对VGG和ResNet的策略进行结合，提出了一种新的split-transform-merge策略，如图所示：

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/1.png)

> 图中左侧是ResNet结构；右边是作者提出的ResNeXt结构，其中引入了一个新维度，称之为cardinality——the size of the set of transformations，图中cardinality=32，即将输入的256通道的影像做了32次相同的操作——1x1卷积  filters=4 —> 3x3卷积  filters=4 —> 1x1卷积  filters=256。

## (二) ResNeXt的block

> 以全连接层为例，全连接层的计算公式和示意图如下：

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/2.png)
![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/3.png)

>上述过程可以很好的表达ResNeXt的split-transform-merge策略即——splitting，transforming，and aggregating。ResNeXt的策略就是将其中的<img src="https://latex.codecogs.com/svg.latex?w_{i}x_{i}" title="w_{i}x_{i}" />替换成更一般的函数，如下所示：

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/4.png)
![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/5.png)

> ResNeXt的block有三种等价形式，如图所示：(a)与图1右边的结构相同；(b)则采用两层卷积之后concatenate，再卷积，有点类似 Inception-ResNet，只不过这里的 paths 都是相同的拓扑结构；(c)采用的是grouped convolutions，这个 group 参数就是 convolusion 层的 group 参数，用来限制本层卷积核和输入 channels 的卷积，最早应该是 AlexNet 上使用，可以减少计算量，这里采用32个 group，每个 group 的输入输出 channels 都是4，最后把channels合并。这三种结构完全等价，作者采用的是(c)结构，因为该结构比较简洁而且速度更快。

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/6.png)

> block的参数如表所示，第一行是cardinality C，第二行是每个path的中间channels数量，第三个是block的宽度，是第一行和第二行的乘积。

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/7.png)

## (三)  与ResNet对比

> ResNeXt50和ResNet50对比如下：

![image](https://github.com/ShaoQiBNU/ResNeXt/blob/master/images/8.png)

> 每个conv中，ResNeXt50的通道数要多于ResNet5，但两者的参数量是一样的，以图1为例，计算二者的参数量如下：

```
ResNet
参数量 = 256 * 64 + 3 * 3 * 64 * 64 + 64 * 256 ≈ 70k

ResNeXt
参数量 = C * (256 * d + 3 * 3 * d * d + d * 256)
图中 C = 32，d = 4，所以参数量 ≈ 70k
```
## (四) 其他
> ResNeXt和ResNet-50/101的区别仅仅在于其中的block，其他都不变，包括BN和ReLU。

# 三. 代码
> 利用MNIST数据集，构建ResNeXt50网络，查看网络效果，由于输入为28 x 28，所以最后的全局池化没有用到，代码如下：
