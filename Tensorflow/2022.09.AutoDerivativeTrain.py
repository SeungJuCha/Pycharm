import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#mnist 데이터 사용하는것이 아님 단순 일차 데이터 사용해야 정확함
mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test)  = mnist.load_data()

print('train set', x_train.shape, y_train.shape)

x_train = x_train/ x_train.max()
x_test = x_test/x_test.max()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    
    tf.keras.layers.Dense(10, activation = 'softmax')
])

loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimiz=  tf.keras.optimizers.Adam()

#기록을 위한 metric 정의
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_acc  = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'train_acc')
valid_loss = tf.keras.metrics.Mean(name ='valid_loss')
valid_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'valid_acc')

#배치 생성 함수

def get_batch(x,y,batch_size = 32):
    for i in range(int(x.shape[0]//batch_size)):
        x_batch = x[i* batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]
        yield(np.asarray(x_batch),np.asarray(y_batch))


"""tf 2.0버전에서는 즉시실행 모드가 default값 바로바로 계산을 진행
    지연 실행모드 = 계산 최적화를 진행(계산 그래프 생성) --> 복잡한 연산 모델일 경우 효율적
    사용 방법 = @tf.function
    이경우 내부에서의 numpy나  파이썬 호출은 전부 상수로 변경되기에 맨 마지막에 사용할것"""

@tf.function
def train_step(images, labels):
    #GradientTape
    with tf.GradientTape() as tape:
        #predict
        prediction = model(images, training = True)
        #loss
        loss =loss_func(labels, prediction)
        #미분 계산
        gradient = tape.gradient(loss, model.trainable_variables)
        #optimizer 적용
        optimiz.apply_gradients(zip(gradient, model.trainable_variables))
        train_loss(loss)
        train_acc(labels,prediction)

@tf.function
def valid_step(images, labels):
    prediction = model(images, training = False)
    loss = loss_func(labels,prediction)

    valid_loss(loss)
    valid_acc(labels,prediction)

#초기화 코드
train_loss.reset_states()
train_acc.reset_states()
valid_loss.reset_states()
valid_acc.reset_states()

#훈련
for epoch in range(5):
    for images,labels in get_batch(x_train,y_train):
        train_step(images,labels)

    for images,labels in get_batch(x_test,y_test):
        valid_step(images,labels)

    metric_template = 'epoch:{},loss:{:.4f},acc:{:.2f}%,val_loss:{:.4f},val_acc:{:.2f}%'
    print(metric_template.format(epoch+1,train_loss.result(),train_acc.result()*100,
                                 valid_loss.result(),valid_acc.result()*100))