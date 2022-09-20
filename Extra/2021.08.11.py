import numpy as np
import pandas as pd

# data = pd.read_csv('D:\\pycharm\\IT innovation\\weather.nominal.csv',sep = ',')
data = pd.read_csv('weather.nominal.csv',sep = ',')
print(data)
print('#'*50)

outlook_tab = pd.crosstab(data['play'],data['outlook'])
print(outlook_tab)
print('#'*50)

Temperature_tab = pd.crosstab(data['play'],data['temperature'])
print(Temperature_tab)
print('#'*50)

Humidity_tab = pd.crosstab(data['play'],data['humidity'])
print(Humidity_tab)
print('#'*50)

Windy_tab = pd.crosstab(data['play'],data['windy'])
print(Windy_tab)
print('#'*50)

Joined_tab = outlook_tab.join(Temperature_tab)
print(Joined_tab)
print('#'*50)

Joined_tab = Joined_tab.join(Humidity_tab)
print(Joined_tab)
print('#'*50)

Joined_tab= Joined_tab.join(Windy_tab)
print(Joined_tab)
print('#'*50)

Joined_tab.iloc[0,:] = Joined_tab.iloc[0,:].apply(lambda  x: x/5)
print(Joined_tab.iloc[0,:])
print('#'*50)

Joined_tab.iloc[1,:] = Joined_tab.iloc[1,:].apply(lambda  x: x/9)
print(Joined_tab.iloc[1,:])
print('#'*50)

print(Joined_tab)

##########################################################
from sklearn import naive_bayes
# naive_bayes_model = naive_bayes()
#naive 에는 여러가지 많은 이벤트 모듈이 존재하기에 특정을 해야함
multinomial_model = naive_bayes.MultinomialNB()


print(data['outlook'])
data = data.apply(lambda  x :x.astype(dtype='category'))
print(data['outlook'])

# multinomial_model.fit(data.iloc[:,:4],data['play'])
#Unable to convert array of bytes/strings into decimal numbers with dtype='numeric'
#에러가 난다 (문자열의 카테고리 학습이 불가능하기 때문) -->숫자로 바꿔야 된다

#dic 형태로 변환
Outlook_dic = {'overcast':0,'rainy':1,'sunny':2}
Temperature_dic = {'cool':0,'hot':1,'mild':2}
Humidity_dic = {'high':0,'normal':1}
Windy_dic ={False:0, True:1} #false와 true는 문자열이아님

data['outlook']= data['outlook'].map(Outlook_dic)
data['temperature'] = data['temperature'].map(Temperature_dic)
data['humidity'] = data['humidity'].map(Humidity_dic)
data['windy'] = data['windy'].map(Windy_dic)

print('Convert str -> number \n',data)


print(multinomial_model.fit(data.iloc[:, :4], data['play']))

print(multinomial_model.predict([[2, 2, 0, 1]]))
#xpected 2D array, got 1D array instead:

print(multinomial_model.predict_proba([[2, 2, 0, 1]]))
#no확률과 yes확률 의 array[]

########################################
#가우시안 모델을 이용하는 연속적인 데이터 처리

from sklearn.datasets import load_iris
iris = load_iris()
iris_df  = pd.DataFrame(iris.data, columns= iris.feature_names)
# print(iris_df.head(3))
iris_df['Species'] = iris.target
# print(iris_df.head(3))

guassian_model = naive_bayes.GaussianNB()
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest = train_test_split(iris_df.iloc[:,:4],
                                             iris_df['Species'],
                                             test_size=0.33)
print(guassian_model.fit(xTrain, yTrain))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(yTest,guassian_model.predict(xTest)))
print(confusion_matrix(yTest,guassian_model.predict(xTest)))


########################
#SVM Support Vector Margin

from sklearn import svm
x = np.array([[0,0],[1,1]])
y = [0,1]
LinearSVM = svm.LinearSVC()
LinearSVM.fit(x, y)
LinearSVM.set_params(penalty = '10')
# 0 ->red spot 1-> blue spot
LinearSVM.predict([[2,2]])
print('기울기 :',LinearSVM.coef_[0])
print('절편 :',LinearSVM.intercept_[0])

import matplotlib.pyplot as plt
from matplotlib import  style
style.use('ggplot')
w = LinearSVM.coef_[0]
b = LinearSVM.intercept_[0]

slope = -w[0]/w[1]
# print(slope)

xx = np.linspace(0,1,5) #0부터 1까지를 5등분
yy = slope * xx - b/w[1]

h0 = plt.plot(xx,yy,'k-',label = 'Hyperplane')
plt.scatter(x[:,0],x[:,1], c = y)
plt.legend()
plt.show()

from sklearn.svm import SVC
style.use('ggplot')
x = np.array([[0,0],[1,1,],[0,1],[1,0]])
y = [0,0,1,1] #0이 파랑, 1이 빨강
xorSVM = SVC()
xorSVM.fit(x,y)

test_data = np.array([[0.8,0.8],[0.2,0.9]])
print('x값 :\n',test_data)
print('판단:',xorSVM.predict(test_data))

np.random.seed(0)
x = np.random.randn(300,2)
y = np.logical_xor(x[:,0]>0,x[:,1]>0)
xorSVM.fit(x,y)

test_data = np.array([[0.8,0.8],[0.2,0.9]])
print('300개 학습 후 판단:',xorSVM.predict(test_data))
test_data = np.array([[0.8,0.8],[0.0,0.9]])
print('300개 학습 후 판단:',xorSVM.predict(test_data))


xx,yy = np.meshgrid(np.linspace(-3,3,500),np.linspace(-3,3,500))
z = xorSVM.decision_function(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.imshow(z, interpolation='nearest',
           extent=(xx.min(),xx.max(),yy.min(),yy.max()),
           aspect='auto',origin='lower',
           cmap = plt.cm.PuOr_r)

contours = plt.contour(xx, yy, z, levels=[0], lw=2, lt='--')
# contours = plt.contour(xx, yy, z, levels=[0], linewidths=2, linetypes='--')

plt.scatter(x[:, 0], x[:, 1], s=30, c=y,cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()


