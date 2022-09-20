import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test)  = mnist.load_data()

print('train set', x_train.shape, y_train.shape)

x_train = x_train/ x_train.max()
x_test = x_test/x_test.max()

model_a =  tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model_b =  tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model_c = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha = 0.2),
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha = 0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model_a.summary()
model_b.summary()
model_c.summary()

model_a.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model_b.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model_c.compile(optimizer = 'SGD',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

# history_a = model_a.fit(x_train,y_train,validation_data = (x_test,y_test),epochs= 10)
# history_b = model_b.fit(x_train,y_train,validation_data = (x_test,y_test),epochs= 10)
# history_c = model_c.fit(x_train,y_train,validation_data = (x_test,y_test),epochs= 10)

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.figure(figsize=(12,9))
# plt.plot(np.arange(1,11),history_a.history['val_loss'],color = 'navy', linestyle = ':')
# plt.plot(np.arange(1,11),history_b.history['val_loss'],color = 'red',linestyle = '-')
# plt.plot(np.arange(1,11),history_c.history['val_loss'],color = 'green',linestyle = '-.')
#
# plt.title('Lo0ssses',fontsize = 20)
# plt.xlabel('epochs')
# plt.ylabel('Losses')
# plt.legend(['ReLU','BatchNorm + ReLU','BatchNorm+LeakyRelu'],fontsize= 12)
# plt.show()

"""Callbacks !!!!"""
#Model Checkpoint epoch별 가중치 저장

# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = 'tmp_checkpoint.ckpt',
#                                                 save_weights_only = True,
#                                                 save_best_only = True,
#                                                 monitor = 'val_loss',
#                                                 verbose = 1)
#
# model_a.fit(x_train,y_train,validation_data = (x_test,y_test),
#             epochs = 10, callbacks = [checkpoint])
#
# loss,acc = model_a.evaluate(x_test,y_test)
# print(f'로드전: loss: {loss:3f}, acc:{acc:3f}')
#
# model_a.load_weights('tmp_checkpoint.ckpt')
# loss,acc = model_a.evaluate(x_test,y_test)
# print(f'로드후:loss: {loss:3f}, acc:{acc:3f}')

# Early stopping 하이퍼파라미터 monitor, min_delta, patience, verbose, mode, basline, restore_best_weights

# earlystopping = tf.keras.callbacks.EarlyStopping(monitor  = 'val_loss', patience = 3)
# model_b.fit(x_train,y_train,validation_data= (x_test,y_test),
#             epochs = 20, callbacks = [earlystopping])

#학습률 스케쥴러 LearningRateScheduler 학습도중 학습률 변경이 필요하다고 판단되는 경우
def scheduler(epoch,lr):
    tf.print(f'learningrate :{lr:.5f}')
    if epoch <=5:
        return lr
    else:
        return lr *tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

print(round(model_c.optimizer.lr.numpy(),5))

model_c.fit(x_train,y_train, validation_data =(x_test,y_test),
            epochs = 10,
            callbacks= [lr_scheduler])


print(round(model_c.optimizer.lr.numpy(),5))