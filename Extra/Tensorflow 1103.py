"""logistic Regression----binary classification (0or 1) 분류"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# x_train = [[1., 2.],
#           [2., 3.],
#           [3., 1.],
#           [4., 3.],
#           [5., 3.],
#           [6., 2.]]
# y_train = [[0.],
#           [0.],
#           [0.],
#           [1.],
#           [1.],
#           [1.]]
#
# x_test = [[5.,2.]]
# y_test = [[1.]]
#
#
# x1 = [x[0] for x in x_train]
# x2 = [x[1] for x in x_train]
#
# #ploting data
# colors = [int(y[0] % 3) for y in y_train]
# plt.scatter(x1,x2, c=colors , marker='^')
# plt.scatter(x_test[0][0],x_test[0][1], c="red")
#
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()
#
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))#.repeat()
#
# #Hypothesis 를 위한 변수 생성
# W = tf.Variable(tf.zeros([2,1]), name='weight')  #y = XW +b 5by1
# b = tf.Variable(tf.zeros([1]), name='bias')
#
# def logistic_regression(features):
#     """sigmoid function"""
#     hypothesis  = tf.divide(1., 1. + tf.exp(-1*(tf.matmul(features, W) + b)))
#     return hypothesis
# def loss_fn(hypothesis, features, labels):  #labels = 실제 Y값  # cost 함수
#     cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) + (1 - labels) * tf.math.log(1 - hypothesis))
#     return cost
# def accuracy_fn(hypothesis, labels):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # cast라는 것은 data type 변경방법
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
#     return accuracy
# def grad(features, labels):
#     with tf.GradientTape() as tape:
#         loss_value = loss_fn(logistic_regression(features),features,labels)
#     return tape.gradient(loss_value, [W,b])
#
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# EPOCHS = 1001
#
# for step in range(EPOCHS):
#     for features, labels  in iter(dataset):
#         grads = grad(features, labels)
#         optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
#         if step % 100 == 0:
#             print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))
# test_acc = accuracy_fn(logistic_regression(x_test),y_test)
# print("Testset Accuracy: {:.4f}".format(test_acc))

"""확장 버전 만약 분류가 여러가지를 해야되는 경우 ex) 동물 등등 label ---> 0,1,2,3,4,.... 증가
    Softmax를 사용"""

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],   #A,B,C 분류로 가능하나 A= 1,0,0 B= 0,1,0 C= 0,0,1
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

#convert into numpy and float format
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

nb_classes = 3 #class의 개수입니다.

#데이터 크기를 알아야 W matrix와 b를 지정이 가능
print(x_data.shape)
print(y_data.shape)

#Hypothesis를 위한 W 와 b를 지정
W = tf.Variable(tf.random.normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random.normal([nb_classes,]), name='bias')
variables = [W, b]

def logit_fn(X):
    return tf.matmul(X, W) + b
def hypothesis(X):
    """soft max = 원래의 x값을 0~1의 사이값에 다 더할경우 1인 확률형태로 변환을 시킨다"""
    return tf.nn.softmax(tf.matmul(X, W) + b)
def cost_fn(X, Y): # cost-function with logits
    """즉 확률을 변수로 받아서 계산을 하겠다 기존의 WX +b의 계산값이 아닌 변환을한 값을 사용"""
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)  # 8by 3의 matrix를 각 요소를 곱해서 평균시 column기준
    cost_mean = tf.reduce_mean(cost)
    return cost_mean
def cost_fn_with_logits(X, Y):
    """cross-entropy방식으로 하는 방법"""
    logits = logit_fn(X)
    cost_i = tf.keras.losses.categorical_crossentropy(y_true=Y, y_pred=logits,
                                                      from_logits=True)
    cost = tf.reduce_mean(cost_i)
    return cost
def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
    return grads
def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % verbose == 0):
            print('Loss at epoch %d: %f' % (i + 1, cost_fn(X, Y).numpy()))


# Test set
sample_db = [[8,2,1,4]]  #1by 4 + 4 by 3 == 1by3
sample_db = np.asarray(sample_db, dtype=np.float32)
a = hypothesis(sample_db)
print(a)
print(tf.argmax(a, axis =1))
b = hypothesis(x_data)
print(b)
print(tf.argmax(b, 1)) # 예측 값
print(tf.argmax(y_data, 1))  # 실제 y값

"""one-hot encoding
    원하는 값을 1 나머지를 전부다 0으로 만드는 방법"""
# y의 값이 0과 1이 아닌 여러다른 수로 이루져있을 경우
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)
