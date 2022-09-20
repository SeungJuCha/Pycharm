
import pandas as pd
import pickle
from sklearn.datasets import load_iris

#저장한 모델파일 읽어오기
f = open('iris_DTmodel.pickle','rb') #rb = read binary
model = pickle.load(f)
f.close()

#분석할 데이터 읽어오기
data = load_iris()
features = pd.DataFrame(data =data.data, columns= data.feature_names)
target = pd.DataFrame(data.target, columns=['species'])
iris = pd.concat([features,target], axis =1)

#학습 없이 예측 그결과 파일 저장
prediction = model.predict(iris.iloc[:,:-1])
iris['predicted_species'] = prediction
iris.to_csv('Report_AI_Species_Iris.csv', index = True)
print(iris)