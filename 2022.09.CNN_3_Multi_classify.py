import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_valid,y_valid)=  mnist.load_data()

#홀짝 label 만들기 정답
y_train_odd = []
for y in y_train:
    if y%2 ==0:
        y_train_odd.append(0)
    else:
        y_train_odd.append(1)

y_train_odd = np.array(y_train_odd)
print(y_train_odd.shape)

#validation set 처리 비교
y_valid_odd =[]
for y in y_valid:
    if y%2 ==0:
        y_valid_odd.append(0)
    else:
        y_valid_odd.append(1)

y_valid_odd = np.array(y_valid_odd)


#normalization of datta
x_train = x_train/255.0
x_valid = x_valid/255.0

#channel 추가 CNN에서 shape은 반드시 가로 세로 두께(칼라 채널)
x_train_in = tf.expand_dims(x_train,-1)
x_valid_in = tf.expand_dims(x_valid,-1)

print(x_valid_in.shape)

"""Functional API-> 다중 출력 2개의outputs 즉 0~9분류 & 홀짝 분류"""

inputs = tf.keras.layers.Input(shape = (28,28,1), name = 'inputs')

conv = tf.keras.layers.Conv2D(filters = 32,kernel_size=(3,3),activation ='relu',
                              name = 'conv2d_layer')(inputs)
pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name = 'pool_layer')(conv)
flat = tf.keras.layers.Flatten(name = 'flatten_layer')(pool)

flat_inputs = tf.keras.layers.Flatten()(inputs)
#합치기 concate
concat = tf.keras.layers.Concatenate()([flat,flat_inputs])

digit_outputs = tf.keras.layers.Dense(units =10, activation ='softmax',
                                      name ='digit_dense')(concat)
odd_outputs = tf.keras.layers.Dense(units = 1, activation = 'sigmoid',
                                    name = 'odd_dense')(flat_inputs)
model= tf.keras.models.Model(inputs =inputs, outputs= [digit_outputs,odd_outputs])


model.summary()

print(model.input)
print(model.output)

# from tensorflow.python.keras.utils.vis_utils import plot_model
# plot_model(model, show_shapes= True,show_layer_names= True)


"""컴파일시 레이어이름을 지정 이 이름이 key, 각 key에 적용할 손실함수와 가중치를 dictionary형태로 지정!!!!"""
model.compile(optimizer = 'adam', loss = {'digit_dense':'sparse_categorical_crossentropy',
                                          'odd_dense': 'binary_crossentropy'},
              loss_weights = {'digit_dense':1, 'odd_dense':0.5},metrics =['accuracy'])

history = model.fit({'inputs':x_train_in},{'digit_dense':y_train,'odd_dense':y_train_odd},
                    validation_data = ({'inputs':x_valid_in},{'digit_dense':y_valid,'odd_dense':y_valid_odd}),
                    epochs = 10)

model.evaluate({'inputs':x_valid_in},{'digit_dense':y_valid,'odd_dense':y_valid_odd})

import matplotlib.pyplot as plt


def image(data, idx):
    plt.figure(figsize=(5, 5))
    plt.imshow(data[idx])
    plt.axis('off')
    plt.show()


image(x_valid, 0)

digit_pred, odd_pred = model.predict(x_valid_in)
print(digit_pred[0])
print(odd_pred[0])

digit_label = np.argmax(digit_pred, axis=-1)

print(digit_label[0:10])

odd_label = (odd_pred > 0.5).astype(np.int).reshape(1, -1)[0]

print(odd_label[0:10])