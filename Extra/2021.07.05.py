import keras as kr
import numpy as np
import tensorflow


print(kr.__version__)

# 설치된 라이브러리 보기
# pip list
print(tensorflow.__version__)
print(np.__version__)

x_train = np.array( [0,1] )
y_train = x_train * 2 + 1
print(x_train,y_train)
x_test = np.array([2,3])
y_test = x_test * 2 + 1
print(x_test, y_test)

model = kr.models. Sequential()
print(type(model))

model. add(kr. layers. Dense(1, input_shape=(1,)))
model. summary()
y_predict = model. predict(x_test)
print(y_predict.flatten())
print(y_test)

model. compile( 'SGD', 'mse')
history = model. fit( x_train, y_train, epochs=1000, verbose=0)
y_predict= model.predict(x_test)
print(y_predict.flatten())
print(y_test)


