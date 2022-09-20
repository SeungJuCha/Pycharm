"""Functional API 특징 다중 입출력 및 복잡한 구조 생성 가능
    마지막에 ()을 치고 전층의 이름을 집어넣어줌으로써 생성가능
    Shape 잘 맞추기!!"""


import tensorflow as tf
import os
import numpy as np
import graphviz
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test)  = mnist.load_data()

print('train set', x_train.shape, y_train.shape)

x_train = x_train/ x_train.max()
x_test = x_test/x_test.max()

input_layer = tf.keras.Input(shape = (28,28),name = 'InputLayer')
x1 = tf.keras.layers.Flatten(name = 'Flatten')(input_layer)
x2 = tf.keras.layers.Dense(256, activation = 'relu', name = 'Dense1')(x1)
x3 = tf.keras.layers.Dense(64, activation = 'relu', name = 'Dense2')(x2)
x4 = tf.keras.layers.Dense(10, activation = 'softmax', name = 'OutputLayer')(x3)

func_model = tf.keras.Model(inputs = input_layer, outputs = x4, name = 'FunctionalModel')

func_model.summary()

from tensorflow.keras.utils import plot_model

plot_model(func_model,show_shapes = True, show_layer_names = True, to_file = 'model.png')