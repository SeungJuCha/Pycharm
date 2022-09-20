import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.keras.layers.Dense(nodes = 10, activation = 'relu')

# """데이터셋"""
# x = np.arange(1,6)
# y = 3*x+2
#
# """model 층 생성"""
# model = keras.Sequential([
#     keras.layers.Dense(1,input_shape = [1])
# ])
# """요약"""
# model.summary()
# """컴파일"""
# model.compile(optimizer = 'sgd',loss = 'mse', metrics =['mae'])
# """훈련및 저장"""
# history = model.fit(x,y,epochs = 1200)
#
# """그림"""
# plt.plot(history.history['loss'],label = 'loss')
# plt.plot(history.history['mae'],label = 'mae')
# plt.xlim(-1,20)
# plt.title('Loss')
# plt.legend()
# plt.show()
# """평가-> 검증용 데이터를 따로 마련 cross-validation으로 진행"""
# model.evaluate(x,y)
# """예측"""
# model.predict([10])

"""MNIST 분류"""
data = tf.keras.datasets.mnist
"""훈련셋과 검증셋으로 분류--> 이미 되어있음"""
(x_train,y_train),(x_test,y_test) = data.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)

"""data picture"""
# fig,axes = plt.subplots(3,5)
# fig.set_size_inches(8,5)
#
# for i in range(15):
#     ax = axes[i//5,i%5]
#     ax.imshow(x_train[i],cmap = 'gray')
#     ax.axis('off')
#     ax.set_title(str(y_train[i]))
#
# plt.tight_layout()
# plt.show()

"""전처리"""
#x_train 배열의 데이터 확인
print(x_train[0, 10:15, 10:15])
#데이터 정규화 max, min 0~255 pixel
# 분포도가 좋아짐
print(x_train.max(), x_train.min())
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
print(x_train[0, 10:15, 10:15])

#Dense lyaer 입력값은 반드시 1차원
# x_train = x_train.reshape(60000,-1)
# print(x_train.shape)
#Flatten layer를 통해 1차원
# x_train = tf.keras.layers.Flatten()(x_train)
# print(x_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(64,activation = 'relu'),
    tf.keras.layers.Dense(32,activation= 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
])

model.summary()

model.compile(loss= 'sparse_categorical_crossentropy',
              optimizer=  tf.keras.optimizers.Adam(learning_rate = 0.001),
              metrics = ['acc'])

history = model.fit(x_train,y_train,validation_data= (x_test,y_test),
                    epochs = 10)

plt.plot(history.history['loss'],color= 'red',label ='loss')
plt.plot(history.history['val_loss'],color ='blue',label= 'val_loss')
plt.plot(history.history['acc'],color = 'green',label = 'acc')
plt.plot(history.history['val_acc'],color= 'yellow',label = 'val_acc')
plt.xlim(-1,20)
plt.title('Loss&Acc')
plt.legend()
plt.show()


test_loss, test_acc=  model.evaluate(x_test,y_test)
print(test_acc)

predictions = model.predict(x_test)
print(np.argmax(predictions[:10],axis= 0))