#Scikit-learn 은 개발자 환경 프레임워크에 최고
#estimator, fit, predict, transform 4가지로 구현 가능

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data,data.target
                                                    ,random_state=42)

model = DecisionTreeClassifier(criterion = 'entropy')
# print(model)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)
#y_pred 와 y_test를 쌍으로 출력(1,0)
#
# for a,b in zip(y_pred,y_test):
#     print((a,b), end = ',')
print()
print('#'*30)
# for i in range(len(y_pred)):
#     if y_pred[i]==y_test[i]:
#         a,b = y_pred[i], y_test[i]
#         print((a,b))
#     else:
#         print('NaN')


from sklearn.preprocessing import StandardScaler
# print(X_train)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # X_train = scaler.fit_transform(X_train) #결합가능
# print(X_train)

#iris데이터셋으로 해보기
from sklearn.datasets import load_iris
data = load_iris()
print(data.target_names)
print(data.keys())
print('#'*30)
# print(data.DESCR)

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

features = pd.DataFrame(data =data.data, columns= data.feature_names)
print(features) #dataframe 생성
print('#'*30)
target = pd.DataFrame(data.target, columns=['species'])
print(target)
print('#'*30)
iris = pd.concat([features,target], axis =1) #concat사용시 여러값을 묶을때는 []로 사용해서 1개로 쓰기
print(iris)
print('#'*30)
iris.rename({'sepal length (cm)':'sepal length',
             'sepal width (cm)':'sepal width',
             'petal length (cm)':'petal length',
             'petal width (cm)':'petal width'},axis = 1, inplace = True)
#inplace = True 는 데이터가 바로 변경되며 업데이트 되었다는 뜻
#기본값은 False형이기에 복사본을 출력
# print(iris)
print('#'*30)
iris['species']= iris.species.map(lambda  x:data.target_names[x])

iris.isna().sum(axis=0)
# print(iris)
print('#'*30)
# print(iris.info())
print('#'*30)
# print(iris.describe())
print('#'*30)
# print(iris.corr())

print(iris.groupby('species').size())

# def boxplot_iris(feature_names, dataset):
#     i = 1
#     plt.figure(figsize=(11,9))
#     for col in feature_names:
#         plt.subplot(2,2,i)  #subplots와는 달리 하나하나 설정해야된다
#         plt.axis('on')  #모든 축 라벨을 켠다+
#         #본래의 default값 (고정값)은 False다
#         plt.tick_params(axis = 'both', left = True, top = False, right = False,
#                         bottom = True, labelleft = True, labeltop = False,
#                         labelright = False, labelbottom = False)
#         #tick_params함수로 눈금의 스타일 설정
#         dataset[col].plot(kind = 'box',subplots = True, sharex =False ,sharey =False)
#         plt.title(col)
#         i +=1
#     plt.show()
#
# boxplot_iris(iris.columns[:-1],iris)


#히스토그램 그리기
# def histogram_iris(feature_names, dataset):
#     i = 1
#     plt.figure(figsize=(11,9))
#     for col in feature_names:
#         plt.subplot(2,2,i)  #subplots와는 달리 하나하나 설정해야된다
#         plt.axis('on')  #모든 축 라벨을 켠다+
#         #본래의 default값 (고정값)은 False다
#         plt.tick_params(axis = 'both', left = True, top = False, right = False,
#                         bottom = True, labelleft = True, labeltop = False,
#                         labelright = False, labelbottom = False)
#         #tick_params함수로 눈금의 스타일 설정
#         dataset[col].hist()
#         plt.title(col)
#         i +=1
#     plt.show()
#
# histogram_iris(iris.columns[:-1],iris)

#heatmp으로 시각화 (피처간의 상관관계)
# corr = iris.corr()
# cmap = sns.diverging_palette(220,10,as_cmap= True)
#
# plt.figure(figsize=(11,9))
# sns.heatmap(corr,cmap= cmap, vmax = 1.0, center = 0, square =True,
#             linewidth =.5,cbar_kws = {'shrink':.5})
# plt.show()
#
# sns.pairplot(iris ,hue = 'species')
# plt.show()
#
# def piechart_iris(feature_names, target, dataset):
#     i = 1
#     plt.figure(figsize=(11,9))
#     for colName in [target]:  #target 0,1,2 3가지가 존재
#         labels = []
#         sizes =[]
#         df = dataset.groupby(colName).size() #묶어서 개수세기
#         # print(df)
#         for key in df.keys(): #
#             labels.append(key) #데이터 네임
#             sizes.append(df[key]) #데이터 값
#         plt.subplot(2,2,i)
#         plt.axis('on')
#         plt.tick_params(axis = 'both',left = False,
#                         top = False, right = False,
#                         bottom = False, labelleft = True,
#                         labeltop = True, labelright = False,
#                         labelbottom = False)
#         plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
#                 #리스트 입력 labels  #autopct는 퍼센트 표시옵션
#                 shadow = True, startangle= 90)  #4사분면 기준 90도에서 start
#         plt.axis('equal')
#         i +=1
#     plt.show()
#
#
# piechart_iris(iris.columns[:-1],iris.species, iris)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.iloc[:,:-1],iris.iloc[:,-1],test_size=0.33, random_state = 42)

# print('*'*80)
# print(X_train)
# print('*'*80)

# print(X_test)
# print('*'*80)
# print(y_test)
# print('*'*80)
# print(y_train)
# print('*'*80)
# print( iris.iloc[:,:-1]) #모든row에서 처음부터 -2까지의 colums출력
# print()
# print(iris.iloc[:,-1]) #처음부터 끝까가지의 row에서 맨 마지막 colums출력
# print(iris)

#알고리즘 선택이 중요----->시행착오 감소
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'gini'
                               ,splitter = 'best'
                               ,max_depth= None, min_samples_split= 2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               max_features=None, random_state= 42,
                               max_leaf_nodes= None, min_impurity_decrease=0.0,
                               min_impurity_split= None, class_weight=None,)
print(model)
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
for pre, test in zip(y_predict,y_test)  :
    print('({},{})'.format(pre,test), end = ',')


print()
#정확도 검증
print(model.score(X_test, y_test))


######교차검증 ##########
import sklearn
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(n_splits= 10, shuffle= True, random_state= 42)
results = cross_val_score(model, X_train,y_train, cv =cv)
fin_result = np.mean(results)

print(results)

# for i, a in enumerate(results): #result반복
#     print('{}번째 교차검증 정확도 :{}'.format(i,a))
# print('\n 교차검증 정확도 :{}'.format(fin_result))

# from sklearn.model_selection import StratifiedKFold
# cv = StratifiedKFold(n_splits= 10, shuffle=True, random_state=42)
# results = cross_val_score(model,X_train,y_train,cv=cv)
# fin_result = np.mean(results)
#
# for i, a in enumerate(results): #result반복
#     print('{}번째 교차검증 정확도 :{}'.format(i,a))
# print('\n 교차검증 정확도 :{}'.format(fin_result))

# import scikitplot as skplt
# skplt.estimators.plot_learning_curve(model, X_train, y_train
#                                      ,figsize =(6,6))
# # plt.show()

from sklearn.model_selection import GridSearchCV
estimator = DecisionTreeClassifier()
parameters = {'max_depth':[4,6,8,10,12], 'criterion': ['gini','entropy']
              ,'splitter':['best','random'], 'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3]
              ,'random_state':[7,23,42,78,142],
              'min_impurity_decrease':[0.0,0.05,0.1,0.2]}
model = GridSearchCV(estimator = estimator, param_grid= parameters, cv=cv,
                     verbose=1, n_jobs= -1, refit = True)

model.fit(X_train,y_train)

print('Best Estimator : \n', model.best_estimator_)
print()
print('Best params :\n', model.best_params_) #최적의 변수
print()
print('Best score: \n',model.best_score_)

#아주 기초적인 분석(데이터가 고루 분포되어있을수록 믿을만하다)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy :',acc)

#예측한 정답 - 실제 정답?
#    예측               정답        오답
#    실제 정답            O(TP)      X(FN)
#    실제 오답            X(FP)      O(TN)
#실제 타겟에 대해 얼마나 맞았는지 다각도 분석이 가능(데이터가 몰려있어도)
from sklearn.metrics import confusion_matrix

confMatrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : \n', confMatrix)
#confusion_matrix(예측데이터, 실제 데이터)

skplt.metrics.plot_confusion_matrix(y_test,y_pred,figsize= (8,6))
# plt.show()

############################################################
#precision(예측클래스중 실제로 맞은 비율)
#precision = (TP) /(TP +FP)

from sklearn.metrics import precision_score
y_precis = precision_score(y_test,y_pred,average= None)
for target, score in zip(data.target_names, y_precis):
    # print('{} 의 정밀도 : {}'.format(target,score))
    print(f'{target}의 정밀도 :{score}')

# recall 타겟 클래스(real값) 중 예측이 맞은 비율 (민감도 sensitivity)
# recall = (TP)/(TP+FN)
from sklearn.metrics import recall_score
y_recall = recall_score(y_test,model.predict(X_test), average= None)
# average 디폴트값은 binary 다중분류일경우는 오류!!
for target, score in zip(data.target_names, y_recall):
    print('{}의 재현율 :{}'.format(target,score))
    #=print(f'{target}의 재현율 :{score}')

###################################
#fall-out 제공하지는 않으나
#타겟이 아닌 실제 클래스중 틀린 비율 1- 특이도(specificity)

##########################
#f-score 정밀도와 재현율의 가중조화평균
#1보다 작을경우 precision에 가중치
#1qhek 클경우 recall에 가중치

from sklearn.metrics import fbeta_score,f1_score

fbetas = fbeta_score(y_test,model.predict(X_test),beta = 0.5,average= None)
for target, score in zip (data.target_names,fbetas):
    print(f'{target}의 f점수 :{score}')

print('='*80)

f1s = f1_score(y_test,model.predict(X_test),average=None)
for target, score in zip (data.target_names, f1s):
    print('{}의 f1점수 ;{}'.format(target,score))

print('='*80)
#Roc curve
#y축 TPR = 참- 에측이 맍은 개수/전체참인 개수
#x축 FPR = 참 - 예측이 틀린 갯수 /전체 거짓인 갯수

#classification_report 전체 일괄 계산
from sklearn.metrics import classification_report

pred = model.predict(X_test)
ClassReport = classification_report(y_test,y_pred)
print('Classification Report :\n',ClassReport)

predProba = model.predict_proba(X_test)
skplt.metrics.plot_roc(y_test,predProba,figsize= (8,6))
plt.show()

model.fit(iris.iloc[:,:-1],iris.iloc[:,-1])
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy :',acc)
#매번 예측할때마다 학습은 비효율
#모델을 저장해두는것이 효율적    pickle를 이용해 모델 객체 생성
#저장방법
import pickle

with open('iris_DTmodel.pickle','wb') as fp :
    pickle.dump(model,fp)

f = open('iris_DTmodel.pickle','rb') #rb = read binary
model = pickle.load(f)
f.close()

prediction = model.predict(iris.iloc[:,:-1])
iris['predicted_species'] = prediction
iris.to_csv('Report_AI_Species_Iris.csv', index = True)



