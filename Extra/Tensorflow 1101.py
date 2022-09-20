import numpy as np
import tensorflow as tf



"""Simple Linear Regression with GradientTape """
"""Hypothesis = W*X + B 변수 한개"""
# X = [1,2,3,4,5]
# Y = [1,2,3,4,5]
#
# # W,B 초기상태 값 지정
# W = tf.Variable(2.9)
# b = tf.Variable(0.5)
# # 학습율 지정
# learning_rate = 0.001
# # W, B update
# for i in range(100):
#     with tf.GradientTape() as tape:  #기울기값 이 0 이되는 min or max 찾기
#         Hypothesis = W * X + b
#         cost = tf.reduce_mean(tf.square(Hypothesis-Y))
#     W_grad, b_grad = tape.gradient(cost,[W,b])
#     W.assign_sub(learning_rate*W_grad)
#     b.assign_sub(learning_rate*b_grad)
#     if i %10 == 0:
#         print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i,W.numpy(),b.numpy(),cost))

"""확장 버전 변수가 많을 경우에 matrix의 곱을 이용한다 y=XW+b"""
data = np.array([
    # X1,   X2,    X3,   y
    [ 73.,  80.,  75., 152. ],
    [ 93.,  88.,  93., 185. ],
    [ 89.,  91.,  90., 180. ],
    [ 96.,  98., 100., 196. ],
    [ 73.,  66.,  70., 142. ]], dtype=np.float32)

#slicing
X_data = data[:,:-1]
y_Data = data[:,-1]

W = tf.Variable(tf.random.normal([3,1]))
print(W)# shape 지정 [a,b]
b= tf.Variable(tf.random.normal([1,]))
print(b)
learning_rate = 0.000001
"""hypothesis,prediction function"""
def predict(X):
    return tf.matmul(X,W)+b

print('epoch|cost')

n_epoch = 2000 # 1000번 시행할 예정
for i in range(n_epoch+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(X_data)-y_Data))

    W_grad, b_grad = tape.gradient(cost, [W,b])
    W.assign_sub(learning_rate*W_grad)
    b.assign_sub(learning_rate*b_grad)

    if i %100 == 0:
        print("{:5}|{:10.4f}".format(i, cost.numpy()))

x_data = [
    [1., 1., 1., 1., 1.], # bias(b)
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]
]
y_data  = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random.uniform((1, 3), -1.0, 1.0)) # [1, 3]으로 변경하고, b 삭제
print(W)

learning_rate = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate)

for i in range(1000+1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, x_data) # b가 없다
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    grads = tape.gradient(cost, [W])
    optimizer.apply_gradients(grads_and_vars=zip(grads,[W]))
    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.4f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], W.numpy()[0][2]))