
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_boston()
xTrain, xTest, yTrain, yTest = \
    train_test_split(data.data, data.target,random_state= 42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xTrain,yTrain)

score = model.score(xTest,yTest)
print('정확도 :',score)


from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split

data = load_diabetes()
xTrain, xTest, yTrain, yTest =\
    train_test_split(data.data, data.target, random_state=42)

#Ridge 형
from sklearn.linear_model import Ridge
ridge = Ridge(alpha= 1.0)
ridge.fit(xTrain, yTrain)
score = ridge.score(xTest,yTest)
print('Ridge 정확도 :',score)

#Lasso 형
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(xTrain,yTrain)
score = lasso.score(xTest,yTest)
print('Lasso 정확도 :',score)

#Elastic Net 형
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=1.0, l1_ratio = 0.5)
#L원(1) 비율임
elastic.fit(xTrain,yTrain)
score = elastic.score(xTest,yTest)
print('Elastic Net 정확도 :',score)

import pandas as pd
import numpy as np

linear = LinearRegression()
linear.fit(xTrain,yTrain)

coefficients = np.vstack((linear.coef_,ridge.coef_,lasso.coef_
                         ,elastic.coef_))
index = ['linear','ridge','lasso','elastic']
coefficients_df = pd.DataFrame(coefficients, columns= data.feature_names,
                               index= index)
print('정규화 선형회구 모델별 가중치 비교')
print(coefficients_df)
# print(coefficients)



#########KNN 모델 전처리 법#############
from sklearn.datasets import load_iris
dataset = load_iris()
data = pd.DataFrame(dataset.data, columns= dataset.feature_names)
xTrain, xTest, yTrain, yTest = train_test_split(data, dataset.target, random_state= 42)

import matplotlib.pyplot as plt

xTrain.plot(kind = 'box')
plt.title('xTrain')
plt.show()
xTest.plot(kind = 'box')
plt.title('xTest')
plt.show()

#정규화를 통해 값의 멈위에 영향을 받지 않도록 전처리
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
xtrain_scaled = mms.fit_transform(xTrain)
xTest_scaled = mms.fit_transform(xTest)
# print(xTrain)
# print(xtrain_scaled)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# 이웃의 개수를 정하는 하이퍼 파라미터는 n_neighbors이다.
# 거리 측정 방법을 정하는 하이퍼 파라미터는 metric이다
model = KNeighborsClassifier(n_neighbors= 5)
model.fit(xtrain_scaled,yTrain)

# KNN 모델의 정확도
yPred = model.predict(xTest_scaled)
print('KNN accuaracy :',accuracy_score(yTest, yPred))

#logistic regression
#성공할 확률을 예측해서 임계치를 넘기면 성공으로 분류하거나, 임계치에 못 미치면 실패로
# 분류하는 이진 분류(Binary Classification) 기능

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

dataset = load_breast_cancer()
train =pd.DataFrame(dataset.data, columns= dataset.feature_names)
target = pd.DataFrame(dataset.target, columns= ['cancer'])
data = pd.concat([train,target],axis= 1)
data.info()

xTrain, xTest, yTrain, yTest = train_test_split(
    data[['mean radius']],data[['cancer']],random_state=42)# mean radius 로 cancer을 예측

model = LogisticRegression(solver= 'liblinear')
model.fit(xTrain,yTrain)

pred = model.predict(xTest)
acc  = accuracy_score(yTest,pred)
print('mean radius 만으로 예측한 결과 :{} \n'.format(acc),pred)

from seaborn import lmplot
lmplot(x = 'mean radius', y = 'cancer', data=data, logistic = True)
plt.show()


#전체 데이터로 학습 및 평가###############
xTrain,xTest,yTrain,yTest = train_test_split(data.loc[:,:'cancer'],
                                             data.loc[:,'cancer'],
                                             random_state= 42)
model.fit(xTrain,yTrain)
yPred = model.predict(xTest)
score = model.score(xTest,yTest)

print(f'전체 데이터로 예측한 결과 :{score}')
print('='*80)


####################################################
import seaborn as sns

data = sns.load_dataset('titanic')
print(data.head(3))
preData = data.drop(columns=['alive','who','adult_male',
                             'class','embark_town'])
preData.drop('deck',axis= 1, inplace= True)
print(preData.head(3))
print('='*80)

preData = preData.dropna().reset_index(drop = True) #데이터 갱신 시켜버리는것 따로 저장 없이
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le =LabelEncoder() #문자열을 숫자로 change 한느법
preData['sex']= le.fit_transform(preData['sex']) #male, female을 0과 1로 변환
print(preData.tail())
print('='*80)

embarkedUniq = preData.embarked.unique()
print('embarked열의 고유한 값: \n',embarkedUniq)
print('='*80)
ohe = OneHotEncoder()
embarked_df = preData[['embarked']]

ohe.fit(embarked_df)
embarked_ohe = ohe.transform(embarked_df)
print(embarked_ohe)
print('='*80)

embarked_df = pd.DataFrame(embarked_ohe.toarray(),columns = embarkedUniq) #array로 변환
print(embarked_df.sample(5))
print('='*80)


pre_data = pd.concat([preData,embarked_df],axis = 1) #합치고
# print(pre_data.iloc[4:12,3:])
pre_data = pre_data.drop('embarked',axis = 1) # enbarked열 삭제
print(pre_data.iloc[4:12,3:])
print('='*80)

xTrain,xTest,yTrain,yTest  =\
    train_test_split(pre_data.iloc[:,1:],pre_data.iloc[:,0],
                     random_state= 42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model. fit (xTrain,yTrain)

score = model.score(xTest,yTest)
print('정확도 :',score)

print('피처 중요도 :\n',model.feature_importances_)
print('='*80)

import matplotlib.font_manager as fm

fm.rcParams['font.family']='Malgun Gothic'
n_feature = xTrain.shape[1]
# print(np.arange(n_feature)) = 10
plt.barh(np.arange(n_feature),model.feature_importances_,
         align= 'center')
plt.yticks(np.arange(n_feature),xTrain.columns)
plt.xlabel('피처 중요도')
plt.ylabel('피처')
plt.ylim(-1,n_feature)

from sklearn.tree import plot_tree

plt.figure(figsize=(8,6))
plot_tree(model.fit(xTrain,yTrain))
plt.show()

# graphviz 라이브러리를 이용하면 좀 더 보기 편하게 시각화
import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(model, out_file= None,
                           feature_names= xTrain.columns,
                           class_names=['사망','생존'],
                           filled= True, rounded= True,
                           special_characters= True)
graph = graphviz.Source(dot_data)
graph #colab에서는 먹힌다