import tensorflow as tf
import numpy as np
import os

"""tf.constant 특징
    내부 자료의 값수정 불가-> np.array와 다른점
    shape 변경은 reshape로 가능"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
a = tf.constant([1,2,3])
b = np.array([1,2,3])

"""간단한 예제"""
g = tf.random.Generator.from_seed(2020)
x = g.normal(shape = (10,))
y = 3*x -2

print('x', x.numpy())
print('y', y.numpy())

"""Loss function"""
def cal_mse(x,y,a,b):
    y_pred = a*x - b
    squared_error = (y-y_pred)**2
    mean_squared_error = tf.reduce_mean(squared_error)
    return mean_squared_error

"""Gradient tape로 미분 과정 기록"""
#변수 지정 with tf.Variable
a = tf.Variable(0.0)
b = tf.Variable(1.0)

Epochs = 200
lr = 0.05
for epoch in range(1,1+Epochs):
    with tf.GradientTape() as tape:
        mse = cal_mse(x,y,a,b)

    grad = tape.gradient(mse,{"a":a,"b":b}) #dictionary형태
    d_a, d_b = grad['a'],grad['b']

    a.assign_sub(d_a*lr)
    b.assign_sub(d_b*lr)

    if epoch%20 ==0:
        print("Epoch %d - MSE: %.4f ----------a: %.2f--b: %.2f" %(epoch,mse,a,b))