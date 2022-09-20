import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("titanic.csv")
C_survived = df[df['생존']==1]['Sex'].value_counts()
print(C_survived)
C_dead = df[df['생존']==0]['Sex'].value_counts()
sex_df = pd.DataFrame([C_survived,C_dead],index = ['Survive','Dead'])
print(sex_df)
sex_df.plot(kind = 'bar',stacked = True)
# plt.show()

#성별의 male 과 female을 0과 1로 change
df['Sex'] = df['Sex'].map({'female':0,'male':1})

# 없는값 채우기 fiilna 결측치 회복

df['Age'] = df['Age'].fillna(value = df['Age'].mean())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_X_feature = df[['Sex','Age','Pclass']]
# data를 0과 1사이의 value를 가지도록 정규화시킨다.
df_X_train = scaler.fit_transform(df_X_feature)
df_y_train = df['생존']

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
model = log.fit(df_X_train,df_y_train)
y_pred = model.predict(df_X_train)
print(model.score(df_X_train,df_y_train))### score 값은 80%


"""누군가의 생존 확률"""
Jack = np.array([0, -0.5924806, 0.82737724]) # male, 22, 3
Rose = np.array([1, -0.8233436520239081, -1.56610693]) # female, 19, 1
print(model.predict_proba((Jack,Rose)))

#K-Nearest_Neighbor  사용
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors= 5, p =2) #p = 거리측정시 L(p) norm 2 유클리디언
knn_model.fit(df_X_train,df_y_train)

print(knn_model.score(df_X_train,df_y_train))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
plt.title("Titanic data set")
ax.scatter(df_X_train[:, 0], df_X_train[:, 1],  df_X_train[:, 2], c = 'red')
# c: marker의 색깔설정, s: marker의 크기
# df_titanic_train[['Sex', 'Age', 'Pclass']]
ax.set_xlabel('Sex')
ax.set_ylabel('Age')
ax.set_zlabel('Pclass')
# plt.show()
# KNN에 적합한 모델일까?


"""SVM  """

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, random_state=3)
print(X[0:5])
print(y[0:5])

def plot_dataset(X, y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1],
                s=20,
                c=y, cmap=plt.cm.Set1)
    plt.axis('tight')
    plt.grid(True, which='both')
    plt.xlim([min(X[:,0])*1.05,max(X[:,0])*1.05])
    plt.ylim([min(X[:,1])*1.05,max(X[:,1])*1.05])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y)
plt.show()



def plot_decision_SVM(X, y, clf):
    plt.figure(figsize=(8, 6))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))   ## 1/||W||
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=150,
                linewidths=1.5,
                edgecolors='green',
                facecolors='none')
    plt.scatter(X[:, 0], X[:, 1],
                s=20,
                c=y, cmap=plt.cm.Set1)
    plt.grid(True, which='both')
    plt.xlim([min(X[:, 0]) * 1.05, max(X[:, 0]) * 1.05])
    plt.ylim([min(X[:, 1]) * 1.05, max(X[:, 1]) * 1.05])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.axis('tight')

    plt.show()

import numpy as np
from sklearn.svm import SVC #Support Vector Classficiation

svm = SVC(kernel='linear', C=1, random_state=1)
svm.fit(X, y)

plot_decision_SVM(X,y,svm)


axis0 = X[:,0] * 10 #X feature만 늘림
print(axis0.shape)
axis1 = X[:,1] * 1
scaled_X = np.array((axis0,axis1)).transpose()
scaled_svm = SVC(kernel='linear', C=1, random_state=1)
scaled_svm.fit(scaled_X, y)

plot_decision_SVM(scaled_X,y,scaled_svm)

from sklearn.preprocessing import StandardScaler
#StandardScaler -> data의 평균이 0 표준편차가 1이 되도록 변환!
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)

std_svm = SVC(kernel='linear', C=1, random_state=1)
std_svm.fit(X_train_std, y)
plot_decision_SVM(X_train_std,y,std_svm)

X, y = make_blobs(n_samples=100, centers=2, random_state=4)  #경계부분에 데이터가 섞여있는 그래프
plot_dataset(X, y)

#### hard margin
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)
# X_train_std = sc.fit_transform(X)
std_svm = SVC(kernel='linear', C=3000, random_state=1) #C값 증가 hard margin
std_svm.fit(X_train_std, y)
plot_decision_SVM(X_train_std,
                  y,
                  std_svm)

###soft margin  C = 1
X, y = make_blobs(n_samples=100, centers=2, random_state=4)

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)
std_svm = SVC(kernel='linear', C=1, random_state=1)
std_svm.fit(X_train_std, y)
plot_decision_SVM(X_train_std,
                  y,
                  std_svm)