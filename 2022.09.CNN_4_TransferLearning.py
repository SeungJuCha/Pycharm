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


#위 모델에서 flatten_layer 출력을 추출
base_model_output = model.get_layer('flatten_layer').output

base_model = tf.keras.models.Model(inputs = model.input, outputs = base_model_output, name = 'base')
base_model.summary()
# 새로운 layer 추가
digit_model = tf.keras.Sequential([
    base_model,tf.keras.layers.Dense(10, activation = 'softmax')
])

digit_model.summary()
#
# digit_model.compile( optimizer = 'adam', loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])
#
# history = digit_model.fit(x_train_in,y_train, validation_data = (x_valid_in,y_valid),
#                           epochs = 10)

"""모델의 파라미터 값을 고정해 훈련을 통한 업데이트 방지 -> frozen _model
    1. base model 전체의 파라미터 고정"""
base_model_frozen = tf.keras.models.Model(inputs = model.input, outputs = base_model_output,
                                          name = 'base_frozen')
base_model_frozen.trainable = False
base_model_frozen.summary()

dense_output = tf.keras.layers.Dense(10, activation = 'softmax')(base_model_frozen.output)
digit_model_frozen = tf.keras.models.Model(inputs = base_model_frozen.input,
                                           outputs = dense_output)

digit_model_frozen.summary()
"""2. 특정 레이어만 파라미터 고정"""

base_model_frozen2 = tf.keras.models.Model(inputs = model.input,
                                           outputs = base_model_output,
                                           name ='base_frozen2')

base_model_frozen2.get_layer('conv2d_layer').trainable = False
base_model_frozen2.summary()