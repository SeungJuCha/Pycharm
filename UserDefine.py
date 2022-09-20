"""사용자 직접 정의 손실함수 사용"""
import numpy as np
import tensorflow as tf
# X = np.array([0.0,1.0,2.0,3.0,4.0,5.0],dtype= float)
# Y = np.array([2.0,4.0,6.0,8.0,10.0,12.0],dtype= float)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units = 1, input_shape = [1])
# ])
#
# #사용자 손실함수 정의 using Huber Loss
# def user_loss(y_true, y_pred):
#     threshold = 1
#     error = y_true-y_pred
#
#     small = tf.abs(error) <=threshold
#     small_error = tf.square(error)/2.0
#     big_error  = threshold*(tf.abs(error)-(threshold/2))
#
#     return tf.where(small,small_error,big_error)
#
# model.compile(optimizer = 'sgd', loss = user_loss)
# model.fit(X,Y,epochs = 1000, verbose = 0)
#
# print(model.predict([6.0]))
#
# """사용자 정의 레이어  tensorflow 제공 layer 클래스를 이용해 내부 수정을 통해 변경 가능"""
# from tensorflow.keras.layers import Layer
#
# class MyDense(Layer):
#     def __init__(self,units = 32, input_shape = None):
#         super(MyDense,self).__init__(input_shape = input_shape)
#         self.units = units
#
#     def build(self,input_shape):
#         #weight 초기화
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(name = 'weight',
#                              initial_value= w_init(shape = (input_shape[-1],self.units),
#                             dtype = 'float32'),trainable= True
#                              )
#         #bias 초기화
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(name = 'bias',
#                              initial_value= b_init(shape = (self.units),dtype = 'float32'),
#                              trainable=True)
#
#     def call(self,inputs):
#         return tf.matmul(inputs, self.w)+self.b
#
#
# model_mylayer = tf.keras.Sequential([
#     MyDense(units = 1, input_shape=[1])
# ])
# model_mylayer.compile(optimizer= 'sgd',loss = 'mse')
# model_mylayer.fit(X,Y)
# print(model.predict([6.0]))

"""train_on_batch"""
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_tset) = mnist.load_data()

x_test = x_test/x_test.max()
x_train = x_train/x_train.max()

model_batch = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(units = 256, activation = 'relu'),
    tf.keras.layers.Dense(units = 64, activation = 'relu'),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 10, activation = 'softmax')
])

Adam = tf.keras.optimizers.Adam(learning_rate = 0.0005)

model_batch.compile(optimizer = Adam, loss ='sparse_categorical_crossentropy',
                    metrics = ['acc'])

print(x_train.shape)
print(x_train.shape[0])
print(x_train.shape[1])
print(60000//32)

"""minibatch 이점 더 자세히 loss판별 가능 단점 시간이 오래걸림"""
"""배치 생성 함수 
60000개의 batch를 32 size로 1875개로 나누기 """
def get_batch(x,y,batch_size =32):
    for i in range(int(x.shape[0]//batch_size)):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]
        yield(np.asarray(x_batch),np.asarray(y_batch))

x,y = next(get_batch(x_train,y_train))
print(x.shape, y.shape)

MONITOR_STEP =50
for epoch in range(1,4):
    batch = 1
    total_loss = 0
    losses = []
    for x,y in get_batch(x_train,y_train,batch_size=128):
        loss, acc = model_batch.train_on_batch(x,y)
        total_loss += loss

        if batch %MONITOR_STEP == 0:
            losses.append(total_loss/batch) # 평균치 loss 50번의 batch당
            print(f'epoch:{epoch},batch:{batch},batch_loss:{loss:.4f},\
                    batch_accuracy:{acc:.4f}, avg_loss:{total_loss/batch:.4f}')
        batch +=1

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(1,batch//MONITOR_STEP+1),losses)
    plt.title(f'epoch:{epoch},losses over batches')
    plt.show()


    loss,acc = model_batch.evaluate(x_test,y_tset)
    print("--"*10)
    print(f'epoch:{epoch},val_loss:{loss:.4f},val_accuracy:{acc:.4f}')
    print()