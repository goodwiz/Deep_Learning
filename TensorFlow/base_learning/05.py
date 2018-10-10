import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data

# 3-2 手写字的介绍

# 载入数据集

mnist = input_data.read_data_sets("/Users/wanghaifeng/PycharmProjects/deep_learning/venv/NN/deep-learning-from-scratch-master/dataset",one_hot= True)




# 每个批次的大小

batch_size = 100
# 计算一共几个批次
n_batch = mnist.train.num_examples

# 命名空间
with tf.name_scope('input'):

    # None 表示批次数量  784表示像素
    x = tf.placeholder(tf.float32,[None,784],name = 'x-input')
    y = tf.placeholder(tf.float32,[None,10],name = 'y-input')

with tf.name_scope('layer'):

    # None 表示批次数量  784表示像素
    x = tf.placeholder(tf.float32,[None,784],name = 'x-input')
    y = tf.placeholder(tf.float32,[None,10],name = 'y-input')


# 创建简单的神经网络

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,w) + b)

# 二次代价函数

loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# 返回预测和真实标签的值是否相同，结果存在一个布尔型列表中，argmax 返回一维向量中最大值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

# 将布尔型转换为浮点型
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(2):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x : batch_xs,y : batch_ys})
        acc = sess.run(accuracy,feed_dict={x : mnist.test.images,y : mnist.test.labels})

        print("Iter " + str(epoch) + " Testing accuracy " + str(acc))
