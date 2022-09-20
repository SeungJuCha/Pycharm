import tensorflow as tf
import os
import numpy as np

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

model_a.summary()

model_a.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

history_a = model_a.fit(x_train,y_train,validation_data = (x_test,y_test),epochs= 10)

test_loss, test_acc=  model_a.evaluate(x_test,y_test)
print(test_acc)

predictions = model_a.predict(x_test)
print(np.argmax(predictions[:10],axis= 0))

#모델 파일저장 및 불러오기

model_a.save('h5-model.h5')
h5_model = tf.keras.models.load_model('h5-model.h5')
h5_model.summary()

loss, acc = h5_model.evaluate(x_test,y_test,verbose = 0)
print(f'h5model] loss:{loss:.5f},acc:{acc:.5f}')



