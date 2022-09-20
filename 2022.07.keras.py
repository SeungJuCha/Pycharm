import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Gradient descent
"""data set"""
def makelinear(w= 0.5, b =0.8, size =50,noise = 1.0):
    x = np.random.rand(size)
    y = w*x +b
    noise =np.random.uniform(-abs(noise), abs(noise), size = y.shape)
    yy = y+noise
    plt.figure(figsize=(10,7))
    plt.plot(x,y,color = 'r',label = ('y = {}x+{}'.format(w,b)))
    plt.scatter(x,yy,label = 'data')
    plt.legend(fontsize = 20)
    plt.show()
    print('w: {},b:{}'.format(w,b))
    return  x,yy

x,y = makelinear(w= 0.3, b =0.5, size =100,noise = 0.01)

num_epoch = 1000

#learning rate
lr = 0.005

#에러기록
errors = []

#random값의로 w,b 초기화
w = np.random.uniform(low = 0.0, high = 1.0)
b = np.random.uniform(low = 0.0, high = 1.0)

for epoch in range(num_epoch):
    y_hat = w*x +b #hypothesis 정의

    #loss function
    error = 0.5*((y_hat - y)**2).sum()
    if error <0.005:
        break
    #Gradient 미분 계산
    w = w-lr*((y_hat-y)*x).sum()
    b=  b-lr*(y_hat-y).sum()
    errors.append(error)

    if epoch %5==0:
        print('{0:2} w = {1:.5f} b = {2:.5f} error = {3:.5f}'.format(epoch,w,b,error))

print("___________"*15)
print('{0:2} w = {1:.1f} b = {2:.1f} error = {3:.5f}'.format(epoch,w,b,error))
# 소수점 조절을 위해 dictionary 형태로 받음

plt.figure(figsize = (10,7))
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

"""다항식---> sum-> mean"""
x1 = np.random.rand(100,)
x2 = np.random.rand(100,)
x3 = np.random.rand(100,)

b = np.random.uniform(low=-1.0, high=1.0)
y = 0.3*x1 +0.5*x2 +0.7*x3 +b

errors = []
w1_grad = []
w2_grad = []
w3_grad = []

num_epoch = 5000
learning_rate = 0.5

#초기화
w1 = np.random.uniform(low=-1.0, high=1.0)
w2 = np.random.uniform(low=-1.0, high=1.0)
w3 = np.random.uniform(low=-1.0, high=1.0)
b = np.random.uniform(low=-1.0, high=1.0)

for epoch in range(num_epoch):
    # 예측값
    y_hat = w1 * x1 + w2 * x2 + w3 * x3 + b
    error = ((y_hat - y) ** 2).mean()
    if error < 0.00001:
        break

    # 미분값 적용 (Gradient)
    w1 = w1 - learning_rate * ((y_hat - y) * x1).mean()
    w2 = w2 - learning_rate * ((y_hat - y) * x2).mean()
    w3 = w3 - learning_rate * ((y_hat - y) * x3).mean()

    w1_grad.append(w1)
    w2_grad.append(w2)
    w3_grad.append(w3)

    b = b - learning_rate * (y_hat - y).mean()

    errors.append(error)

    if epoch % 5 == 0:
        print("{0:2} w1 = {1:.5f}, w2 = {2:.5f}, w3 = {3:.5f}, b = {4:.5f} error = {5:.5f}".format(epoch, w1, w2, w3, b,
                                                                                                   error))

print("----" * 15)
print("{0:2} w1 = {1:.1f}, w2 = {2:.1f}, w3 = {3:.1f}, b = {4:.1f} error = {5:.5f}".format(epoch, w1, w2, w3, b, error))

#가중치 변화 확인
plt.figure(figsize=(10, 7))

plt.hlines(y=0.3, xmin=0, xmax=len(w1_grad), color='r')
plt.plot(w1_grad, color='g')
plt.ylim(0, 1)
plt.title('W1', fontsize=16)
plt.legend(['W1 Change', 'W1'])
plt.show()