import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=  mnist.load_data()

print(x_train.shape, y_train.shape)

import matplotlib.pyplot as plt
def plot_imates(data,idx):
    plt.figure(figsize=(5,5))
    plt.imshow(data[idx],cmap = 'gray')
    plt.axis('off')
    plt.show()

plot_imates(x_train,0)

#data 전처리를 위한 0~1 정규화
print(x_train.min(),x_train.max())

x_train = x_train/255.0
x_test = x_test/255.0
"""mnist = 흑백 데이터 칼라 추가 채널 생성
28by28그림 60000개가 있는데 칼라 채널 생성"""

print(x_train.shape, x_test.shape)
x_train_in = x_train[...,tf.newaxis]
x_test_in =  x_test[...,tf.newaxis]
print(x_train_in.shape, x_test_in.shape)

CNN_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size = (3,3),strides = 1,activation =  'relu',
                           input_shape =(28,28,1),name = 'conv'),
    tf.keras.layers.MaxPooling2D((2,2),name = 'pool'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 10, activation = 'softmax')
])

CNN_model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
history = CNN_model.fit(x_train_in,y_train,
                        validation_data = (x_test_in,y_test),
                        epochs = 10)

CNN_model.evaluate(x_test_in,y_test)

def plot_loss_acc(history,epoch):
    loss,val_loss = history.history['loss'],history.history['val_loss']
    acc,val_acc = history.history['accuracy'],history.history['val_accuracy']

    fig,axes = plt.subplots(1,2,figsize =(12,4))
    axes[0].plot(range(1,1+epoch),loss, label ='training')
    axes[0].plot(range(1, 1 + epoch), val_loss, label='test')
    axes[0].legend(loc = 'best')
    axes[0].set_title("Loss")
    axes[1].plot(range(1, 1 + epoch), acc, label='training')
    axes[1].plot(range(1, 1 + epoch), val_acc, label='test')
    axes[1].legend(loc = 'best')
    axes[1].set_title("ACC")
    plt.show()

plot_loss_acc(history,10)