########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)


########## set net hyperparameters ##########
learning_rate=0.0001

epochs=20
batch_size_train=128
batch_size_test=100

display_step=20


##### cardinality #####
cardinality = 32

########## set net parameters ##########
#### img shape:28*28 ####
n_input=784 

#### 0-9 digits ####
n_classes=10

# Handle Dimension Ordering for different backends
'''
img_input_shape=(224, 224, 3)
concat_axis = 3

img_input_shape=(3, 224, 224)
concat_axis=1
'''
global concat_axis

concat_axis=3

########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


##################### build net model ##########################

######### identity_block #########
def identity_block(inputs,filters,kernel,strides):
    '''
    identity_block: 三层的恒等残差块，影像输入输出的height和width保持不变，channel发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 三层恒等残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3=filters
    k1, k2, k3=kernel
    s1, s2, s3=strides


    ######## shortcut 第一种规则，影像输入输出的height和width保持不变，输入直接加到卷积结果上 ########
    inputs_shortcut=inputs


    ######## first identity block 第一层恒等残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second identity block 第二层恒等残差块 ########
    #### conv ####
    #### 初始输入备份 ####
    concat_feat=layer1

    for i in range(cardinality):

        #### conv 卷积运算 3 x 3 ####
        layer2=tf.layers.conv2d(layer1,filters=int(f2/cardinality),kernel_size=k2,strides=s2,padding='SAME')

        #### 连接 ####
        concat_feat=tf.concat([concat_feat,layer2], concat_axis)

    #### BN ####
    layer2=tf.layers.batch_normalization(concat_feat)

    #### relu ####
    layer2=tf.nn.relu(layer2)


    ######## third identity block 第三层恒等残差块 ########
    #### conv ####
    layer3=tf.layers.conv2d(layer2,filters=f3,kernel_size=k3,strides=s3,padding='SAME')

    #### BN ####
    layer3=tf.layers.batch_normalization(layer3)


    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,layer3)


    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######## convolutional_block #########
def convolutional_block(inputs,filters,kernel,strides):
    '''
    convolutional_block: 三层的卷积残差块，影像输入输出的height、width和channel均发生变化

    inputs: input x 输入影像
    filters: convlution filters 卷积核个数
    kernel: convlution kernel size 卷积核大小
    strides: convlution stride 卷积步长

    return: out 三层的卷积残差块输出
    '''

    ######## conv parameter 卷积参数 ########
    f1, f2, f3=filters
    k1, k2, k3=kernel
    s1, s2, s3=strides


    ######## shortcut 第二种规则，影像输入输出height和width发生变化，需要对输入做调整 ########
    #### conv ####
    inputs_shortcut=tf.layers.conv2d(inputs,filters=f3,kernel_size=1,strides=s1,padding='SAME')

    #### BN ####
    inputs_shortcut=tf.layers.batch_normalization(inputs_shortcut)


    ######## first convolutional block 第一层卷积残差块 ########
    #### conv ####
    layer1=tf.layers.conv2d(inputs,filters=f1,kernel_size=k1,strides=s1,padding='SAME')

    #### BN ####
    layer1=tf.layers.batch_normalization(layer1)

    #### relu ####
    layer1=tf.nn.relu(layer1)


    ######## second convolutional block 第二层卷积残差块 ########
    #### conv ####
    #### 初始输入备份 ####
    concat_feat=layer1

    for i in range(cardinality):
        
        #### conv 卷积运算 3 x 3 ####
        layer2=tf.layers.conv2d(layer1,filters=int(f2/cardinality),kernel_size=k2,strides=s2,padding='SAME')

        #### 连接 ####
        concat_feat=tf.concat([concat_feat,layer2], concat_axis)

    #### BN ####
    layer2=tf.layers.batch_normalization(layer2)

        #### relu ####
    layer2=tf.nn.relu(layer2)


    ######## third convolutional block 第三层卷积残差块 ########
    #### conv ####
    layer3=tf.layers.conv2d(layer2,filters=f3,kernel_size=k3,strides=s3,padding='SAME')

    #### BN ####
    layer3=tf.layers.batch_normalization(layer3)


    ######## shortcut connection 快捷传播：卷积结果+输入 ########
    out=tf.add(inputs_shortcut,layer3)


    ######## relu ########
    out=tf.nn.relu(out)
    
    return out


######### Resnxet 50 layer ##########
def Resnext50(x,n_classes):

    ####### reshape input picture ########
    x=tf.reshape(x,shape=[-1,28,28,1])


    ####### first conv ########
    #### conv ####
    conv1=tf.layers.conv2d(x,filters=64,kernel_size=7,strides=2,padding='SAME')

    #### BN ####
    conv1=tf.layers.batch_normalization(conv1)

    #### relu ####
    conv1=tf.nn.relu(conv1)


    ####### max pool ########
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


    ####### second conv ########
    #### convolutional_block 1 ####
    conv2=convolutional_block(pool1,filters=[128,128,256],kernel=[1,3,1],strides=[1,1,1])

    #### identity_block 2 ####
    conv2=identity_block(conv2,filters=[128,128,256],kernel=[1,3,1],strides=[1,1,1])
    conv2=identity_block(conv2,filters=[128,128,256],kernel=[1,3,1],strides=[1,1,1])


    ####### third conv ########
    #### convolutional_block 1 ####
    conv3=convolutional_block(conv2,filters=[256,256,512],kernel=[1,3,1],strides=[2,1,1])

    #### identity_block 3 ####
    conv3=identity_block(conv3,filters=[256,256,512],kernel=[1,3,1],strides=[1,1,1])
    conv3=identity_block(conv3,filters=[256,256,512],kernel=[1,3,1],strides=[1,1,1])
    conv3=identity_block(conv3,filters=[256,256,512],kernel=[1,3,1],strides=[1,1,1])


    ####### fourth conv ########
    #### convolutional_block 1 ####
    conv4=convolutional_block(conv3,filters=[512,512,1024],kernel=[1,3,1],strides=[2,1,1])
    
    #### identity_block 5 ####
    conv4=identity_block(conv4,filters=[512,512,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[512,512,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[512,512,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[512,512,1024],kernel=[1,3,1],strides=[1,1,1])
    conv4=identity_block(conv4,filters=[512,512,1024],kernel=[1,3,1],strides=[1,1,1])


    ####### fifth conv ########
    #### convolutional_block 1 ####
    conv5=convolutional_block(conv4,filters=[1024,1024,2048],kernel=[1,3,1],strides=[2,1,1])
    
    #### identity_block 2 ####
    conv5=identity_block(conv5,filters=[1024,1024,2048],kernel=[1,3,1],strides=[1,1,1])
    conv5=identity_block(conv5,filters=[1024,1024,2048],kernel=[1,3,1],strides=[1,1,1])


    ####### 全局平均池化 ########
    #pool2=tf.nn.avg_pool(conv5,ksize=[1,7,7,1],strides=[1,7,7,1],padding='VALID')


    ####### flatten 影像展平 ########
    flatten = tf.reshape(conv5, (-1, 1*1*2048))


    ####### out 输出，10类 可根据数据集进行调整 ########
    out=tf.layers.dense(flatten,n_classes)


    ####### softmax ########
    out=tf.nn.softmax(out)

    return out


########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred=Resnext50(x,n_classes)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size_train):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size_train)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size_test):
        batch_x,batch_y=mnist.test.next_batch(batch_size_test)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))