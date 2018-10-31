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

```python
########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_sets", one_hot=True)

########## set net hyperparameters ##########
learning_rate = 0.0001

epochs = 20
batch_size_train = 128
batch_size_test = 100

display_step = 20

##### cardinality #####
cardinality = 32

########## set net parameters ##########
#### img shape:28*28 ####
n_input = 784

#### 0-9 digits ####
n_classes = 10

# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis = 3

########## placeholder ##########
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


##################### build net model ##########################

######### identity_block #########
def identity_block(inputs, filters, kernel, strides):
    '''
    identity_block: 三层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 三层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3 = filters
    k1, k2, k3 = kernel
    s1, s2, s3 = strides

    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut = inputs

    ######## first identity block 第一层恒等残差块 ########
    #### conv ####
    layer1 = tf.layers.conv2d(inputs, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    #### 初始输入备份 ####
    concat_feat = layer1

    for i in range(cardinality):
        #### conv 卷积运算 3 x 3 ####
        layer2 = tf.layers.conv2d(layer1, filters=int(f2 / cardinality), kernel_size=k2, strides=s2, padding='SAME')

        #### 连接 ####
        concat_feat = tf.concat([concat_feat, layer2], concat_axis)

    #### BN ####
    layer2 = tf.layers.batch_normalization(concat_feat)

    #### relu ####
    layer2 = tf.nn.relu(layer2)

    ######## third identity block 第三层恒等残差块 ########
    #### conv ####
    layer3 = tf.layers.conv2d(layer2, filters=f3, kernel_size=k3, strides=s3, padding='SAME')

    #### BN ####
    layer3 = tf.layers.batch_normalization(layer3)

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer3)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######## convolutional_block #########
def convolutional_block(inputs, filters, kernel, strides):
    '''
    convolutional_block: 三层的卷积残差块，影像输入输出的height、width和channel均发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 三层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3 = filters
    k1, k2, k3 = kernel
    s1, s2, s3 = strides

    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut = tf.layers.conv2d(inputs, filters=f3, kernel_size=1, strides=s1, padding='SAME')

    #### BN ####
    inputs_shortcut = tf.layers.batch_normalization(inputs_shortcut)

    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1 = tf.layers.conv2d(inputs, filters=f1, kernel_size=k1, strides=s1, padding='SAME')

    #### BN ####
    layer1 = tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1 = tf.nn.relu(layer1)

    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    #### 初始输入备份 ####
    concat_feat = layer1

    for i in range(cardinality):
        #### conv 卷积运算 3 x 3 ####
        layer2 = tf.layers.conv2d(layer1, filters=int(f2 / cardinality), kernel_size=k2, strides=s2, padding='SAME')

        #### 连接 ####
        concat_feat = tf.concat([concat_feat, layer2], concat_axis)

    #### BN ####
    layer2 = tf.layers.batch_normalization(layer2)

    #### relu ####
    layer2 = tf.nn.relu(layer2)

    ######## third convolutional block 第三层卷积残差块 ########
    #### conv ####
    layer3 = tf.layers.conv2d(layer2, filters=f3, kernel_size=k3, strides=s3, padding='SAME')

    #### BN ####
    layer3 = tf.layers.batch_normalization(layer3)

    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out = tf.add(inputs_shortcut, layer3)

    ######## relu ########
    out = tf.nn.relu(out)

    return out


######### Resnxet 50 layer ##########
def Resnext50(x, n_classes):
    ####### reshape input picture ########
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    ####### first conv ########
    #### conv ####
    conv1 = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='SAME')

    #### BN ####
    conv1 = tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1 = tf.nn.relu(conv1)

    ####### max pool ########
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    ####### second conv ########
    #### convolutional_block 1 ####
    conv2 = convolutional_block(pool1, filters=[128, 128, 256], kernel=[1, 3, 1], strides=[1, 1, 1])

    #### identity_block 2 ####
    conv2 = identity_block(conv2, filters=[128, 128, 256], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv2 = identity_block(conv2, filters=[128, 128, 256], kernel=[1, 3, 1], strides=[1, 1, 1])

    ####### third conv ########
    #### convolutional_block 1 ####
    conv3 = convolutional_block(conv2, filters=[256, 256, 512], kernel=[1, 3, 1], strides=[2, 1, 1])

    #### identity_block 3 ####
    conv3 = identity_block(conv3, filters=[256, 256, 512], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv3 = identity_block(conv3, filters=[256, 256, 512], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv3 = identity_block(conv3, filters=[256, 256, 512], kernel=[1, 3, 1], strides=[1, 1, 1])

    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4 = convolutional_block(conv3, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[2, 1, 1])

    #### identity_block 5 ####
    conv4 = identity_block(conv4, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv4 = identity_block(conv4, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv4 = identity_block(conv4, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv4 = identity_block(conv4, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv4 = identity_block(conv4, filters=[512, 512, 1024], kernel=[1, 3, 1], strides=[1, 1, 1])

    ####### fifth conv ########
    #### convolutional_block 1 ####
    conv5 = convolutional_block(conv4, filters=[1024, 1024, 2048], kernel=[1, 3, 1], strides=[2, 1, 1])

    #### identity_block 2 ####
    conv5 = identity_block(conv5, filters=[1024, 1024, 2048], kernel=[1, 3, 1], strides=[1, 1, 1])
    conv5 = identity_block(conv5, filters=[1024, 1024, 2048], kernel=[1, 3, 1], strides=[1, 1, 1])

    ####### 全局平均池化 ########
    # pool2=tf.nn.avg_pool(conv5,ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')

    ####### flatten 影像展平 ########
    flatten = tf.reshape(conv5, (-1, 1 * 1 * 2048))

    ####### out 输出，10类 可根据数据集进行调整 ########
    out = tf.layers.dense(flatten, n_classes)


    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred = Resnext50(x, n_classes)

#### loss 损失计算 ####
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    #### epoch 世代循环 ####
    for epoch in range(epochs + 1):

        #### iteration ####
        for _ in range(mnist.train.num_examples // batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y = mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            ##### show loss and acc #####
            if step % display_step == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch Loss=" + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples // batch_size_test):
        batch_x, batch_y = mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
```
