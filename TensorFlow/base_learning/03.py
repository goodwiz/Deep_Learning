import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data

# 3-1 非线性回归

# 随机生成100个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个占位符
x = tf.placeholder(tf.float32,[None,1]) #只有一列，根据样本定义
y = tf.placeholder(tf.float32,[None,1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(20000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value,'r-',lw = 6)
    plt.savefig('./S_2000.jpg')
    plt.show()

# 3-2 手写字的介绍

# 载入数据集

mnist = input_data.read_data_sets("/Users/wanghaifeng/PycharmProjects/deep_learning/venv/NN/deep-learning-from-scratch-master/dataset",one_hot= True)


# 每个批次的大小

batch_size = 100
# 计算一共几个批次
n_batch = mnist.train.num_examples
# None 表示批次数量  784表示像素
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

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
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x : batch_xs,y : batch_ys})
        acc = sess.run(accuracy,feed_dict={x : mnist.test.images,y : mnist.test.labels})

        print("Iter " + str(epoch) + " Testing accuracy " + str(acc))


