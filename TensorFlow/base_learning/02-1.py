import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data

# 2-1 创建图和启动图

m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

product = tf.matmul(m1,m2)

# print(product)

sess = tf.Session()
# sess的run方法执行矩阵乘法
result = sess.run(product)
# print(result)

with tf.Session() as sess:
    result = sess.run(product)
    # print(result)

# 2-2 变量的使用

x = tf.Variable([1,2])
a = tf.constant([3,3])
# 创建一个减法和加法
sub = tf.subtract(x,a)
add = tf.add(x,a)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

state = tf.Variable(0,name = 'counter')
new_value = tf.add(state,1)
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))


# 2-3 fetch and feed

# fecth 同时运行多个对话
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input3,input2)
mul = tf.multiply(input1 ,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)

# feed
# 创建占位符

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    # feed的数据以字典的形式传入
    print(sess.run(output,feed_dict =
    {input1:[8,],input2:[2,]}))

# 2-4 简单实例


# 使用numpy 随机生成100个点
x_data =np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss =tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降来进行训练优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 ==0:
            print(step, sess.run([k,b]))








