# import tensorflow.keras as kr
# import numpy as np
#
# x_train = np.array( [0,1] )
# y_train = np.array([1,3])
# print(x_train, y_train)
# x_test = np.array([2,3])
# y_test = np.array([5,7])
# print(x_test, y_test)
#
# x = kr.layers.Input(shape=(1,))
# print(type(x))
# d1 = kr.layers.Dense(2)(x)
# y = kr.layers.Dense(1)(d1)
# print(type(y))
# # d2 = d(x)
# print(type(y))
# print(type(d1))
#
# model = kr.models.Model(x, y)
# model.compile('SGD', 'mse') #Optimizer, loss(cost function)
# history = model.fit( x_train, y_train, epochs = 1000, verbose = 1 )
#
# y_predict = model.predict(x_test)
# print(y_predict.flatten())
# print(y_test)



import matplotlib.pyplot as plt;
import numpy as np;
import tensorflow as tf
(images_train, labels_train), (images_test, labels_test) \
    = tf.keras.datasets.mnist.load_data()
print('훈련데이터 라벨 [{}]: \n'.format(labels_train.shape))
print('훈련데이터 이미지[{}] : \n'.format(images_train.shape))
# 모델 생성, 레이어 2개, 이미지 사이즈 28*28

model = tf.keras.Sequential(layers=[
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(images_train, labels_train, epochs=5)

test_loss, test_acc = model.evaluate(images_train,labels_train,verbose=2)
print("\n테스트 정확도 :", test_acc)

predictions = model.predict(images_test)
pred = np.argmax(predictions[0])
print("예측값 : {}, 실제값 : {}".format(pred, labels_test[0]))

plt.imshow(images_test[0]);
print(history.history)
plt.show();

plt.plot(history.history["loss"])
plt.plot(history.history["accuracy"])
plt.legend(["loss", "accuracy"])

plt.show();


################### Auto Encoder #########################
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
#
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Model
# (x_train, _), (x_test, _) = fashion_mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
#
# print (x_train.shape)
# print (x_test.shape)
# latent_dim = 64
#
#
# class Autoencoder(Model):
#     def __init__(self, encoding_dim):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([
#             layers.Flatten(),
#             layers.Dense(latent_dim, activation='relu'),
#         ])
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(784, activation='sigmoid'),
#             layers.Reshape((28, 28))
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# autoencoder = Autoencoder(latent_dim)
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())# autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
# encoded_imgs = autoencoder.encoder(x_test).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#   # display original
#   ax = plt.subplot(2, n, i + 1)
#   plt.imshow(x_test[i])
#   plt.title("original")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
#
#   # display reconstruction
#   ax = plt.subplot(2, n, i + 1 + n)
#   plt.imshow(decoded_imgs[i])
#   plt.title("reconstructed")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# plt.show()
#
# (x_train, _), (x_test, _) = fashion_mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train[..., tf.newaxis];     x_test = x_test[..., tf.newaxis]
# print(x_train.shape)
#
# noise_factor = 0.2
# x_train_noisy\
#     = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
# x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
#
# x_train_noisy \
#     = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
# x_test_noisy \
#     = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
#
# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.title("original + noise")
#     plt.imshow(tf.squeeze(x_test_noisy[i]))
#     plt.gray()
# plt.show()