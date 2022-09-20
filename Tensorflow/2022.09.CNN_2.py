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

"""Functional API"""

inputs = tf.keras.layers.Input(shape = (28,28,1))

conv = tf.keras.layers.Conv2D(filters = 32,kernel_size=(3,3),activation ='relu')(inputs)
pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)
flat = tf.keras.layers.Flatten()(pool)

flat_inputs = tf.keras.layers.Flatten()(inputs)
#합치기 concate
concat = tf.keras.layers.Concatenate()([flat,flat_inputs])
outputs = tf.keras.layers.Dense(units =10, activation ='softmax')(concat)

model= tf.keras.models.Model(inputs =inputs, outputs= outputs)

model.summary()

# from tensorflow.python.keras.utils.vis_utils import plot_model
# plot_model(model, show_shapes= True,show_layer_names= True)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])

history = model.fit(x_train,y_train, validation_data =(x_valid,y_valid),epochs = 10)

val_loss,val_acc = model.evaluate(x_valid_in,y_valid)
print(val_acc,val_loss)