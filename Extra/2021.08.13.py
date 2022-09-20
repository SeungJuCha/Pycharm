#K-means Clustering 생성

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data #값만 지정하는것 분류명은 no
data_pd = pd.DataFrame(data=data, columns= iris.feature_names)
xTrain, xTest, yTrain, yTest = train_test_split(data, iris.target,
                                                random_state= 42)
# print(data_pd.head(3))

change_col = {'sepal length (cm)': 'sepal length','sepal width (cm)':'sepal width'
              ,'petal length (cm)': 'petal length','petal width (cm)':'petal width'}

data_pd.rename(change_col, axis = 1, inplace=True)
#inplace =True함으로써 내용 업데이트 해야함
print(data_pd.head(3))

petals = pd.DataFrame(data_pd.values[:,2:4],columns=['petal length','petal width'])

plt.scatter(petals.values[:,0],petals.values[:,1])
plt.show()

#Clustering 과정

from sklearn.cluster import KMeans

plt.figure(figsize=(7,5))
km = KMeans(n_clusters= 2, random_state= 20)
km.fit(data_pd.iloc[:,2:4])
y_pred = km.predict(data_pd.iloc[:,2:4])
print(y_pred)
plt.scatter(data_pd.iloc[:,2],data_pd.iloc[:,3],c = y_pred)
plt.title('Clustering')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show() #기서 잠시점하나가 상당히 다른 모습을 확인

#다른 값만 한번 보자
print(data_pd.iloc[98,2:4])
#기준값의 거리에 따라 판단했을 거기에 기준값 확인
print(km.cluster_centers_)

#이 기준치로부터 유클리드 거리 측정

def distance(x1,y1,x2,y2):
    dx = x2-x1
    dy = y2-y1
    squared = dx**2 + dy**2
    result = math.sqrt(squared)
    return result

print('0 cluster distane : \n',distance(data_pd.iloc[98,2],
                                        data_pd.iloc[98,3]
                                        ,km.cluster_centers_[0][0]
                                        ,km.cluster_centers_[0][1]))
print('1 cluster distane : \n',distance(data_pd.iloc[98,2],
                                        data_pd.iloc[98,3]
                                        ,km.cluster_centers_[1][0]
                                        ,km.cluster_centers_[1][1]))
#0번 cluster 기준값이 1번 보다 가까운것으로 판단됬기에 이상한 점이 보인것

### cluster 갯수를 늘리면서 프린팅
n_cluster = [3,4,6,12]
for i in n_cluster:
    count = 1
    km = KMeans(n_clusters= i, random_state= 20)

    km.fit(data_pd.iloc[:,2:4])
    y_pred = km.predict(data_pd.iloc[:,2:4])
    plt.figure(count)
    plt.scatter(data_pd.iloc[:,2],data_pd.iloc[:,3],c = y_pred)
    plt.title('Clustering'+ str(i))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    count+=1
    plt.show()

#####################################################
# 8 개의 Cluster들을 각 영역별로 쪼개어 표시 할 수 있는 Voronoi 그래프로 한번 묘사

km12 = KMeans(n_clusters= 8,random_state= 20)
km12.fit(data_pd.iloc[:,2:4])
y_pred12 = km12.predict(data_pd.iloc[:,2:4])
plt.title('Clustering at 8')
plt.xlabel('petal length')
plt.ylabel('petal width')
graph1 = plt.scatter(data_pd.iloc[:,2],data_pd.iloc[:,3]
                     ,c = y_pred12)
plt.show()

h = .02
km12 = KMeans(n_clusters= 8,random_state= 20)
km12.fit(data_pd.iloc[:,2:4])
y_pred12 = km12.predict(data_pd.iloc[:,2:4])
x_min,x_max = data_pd.iloc[:,2].min() - 1, data_pd.iloc[:,2].max()+1
y_min,y_max = data_pd.iloc[:,3].min() - 1, data_pd.iloc[:,3].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

Z = km12.predict(np.c_[xx.ravel(),yy.ravel()])

Z=Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation= 'nearest',extent=(xx.min(),xx.max()
                                               ,yy.min(),yy.max()),
           cmap=plt.cm.Paired, aspect='auto',origin='lower')
plt.plot(data_pd.iloc[:,2],data_pd.iloc[:,3],'bo',markersize = 2)

centroides = km12.cluster_centers_
plt.scatter(centroides[:,0], centroides[:,1]
            ,marker=  '^',s = 16, linewidths= 3,
            color = 'r',zorder = 10)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#########################################################
# Hierarchical Clustering/ Agglomerative 관점을사용(상향식)



from sklearn.cluster import AgglomerativeClustering
linkage = ['complete','average','ward']
for idx, i in enumerate(linkage):
    plt.figure(idx)
    hier = AgglomerativeClustering(n_clusters=3,affinity='euclidean'
                                   ,linkage= i)
    hier.fit(data_pd.iloc[:,2:4])
    plt.scatter(data_pd.iloc[:,2],data_pd.iloc[:,3],
                c=hier.labels_)
    plt.title('Clustering_'+i)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
plt.show()

#어떤식으로 Tree가 되어있는지 확인
# from scipy.cluster import hierarchy
# hierar = hierarchy.linkage(data_pd.iloc[:,2:4],'complete')
# plt.figure(figsize=(20,20))
# dn = hierarchy.dendrogram(hierar)
#
#
# hierar = hierarchy.linkage(data_pd.iloc[:,2:4],'single')
# plt.figure(figsize=(20,20))
# dn = hierarchy.dendrogram(hierar)

from sklearn.cluster import DBSCAN
# db = DBSCAN(eps = 0.5, min_samples=10)
db = DBSCAN(eps = 0.5, min_samples=5)
db.fit(data_pd.iloc[:,2:4])
y_pred = db.fit_predict(data_pd.iloc[:,2:4])
plt.scatter(data_pd.iloc[:,2],data_pd.iloc[:,3],c = y_pred)
plt.show()
print('IRIS labels \n',db.labels_)

