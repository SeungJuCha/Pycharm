def factorial(n):
    if n==1:
        return 1
    else:
        return n* factorial(n-1)

print(factorial(5))

import numpy as np
la1 = np.array([1,2])
la2 = np.array([3,4])
#
# la1 = [1,2]
# la2 = [3,4]

print(la1+la2)

arr = np.array([[0,1,1,1,1],[0,0,1,1,1],[0,0,0,1,1]])
print(arr.reshape(5, 3))
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.05):

    # marker와 colormap 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #  decision region 가시화
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # 테스트 샘플을 부각하여 그립니다.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1.5,
                    marker='o',
                    s=100,
                    label='test set')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_moons
x,y = make_moons(n_samples=100, noise=0.15, random_state= 42 )
linear_kernel_svm_clf = Pipeline([('scaler',StandardScaler()),('svm_clf',SVC(kernel='poly',degree= 1, coef0=1,C=1))])

linear_kernel_svm_clf.fit(x,y)

plot_decision_regions(x,y,linear_kernel_svm_clf)
plt.show()

from sklearn import metrics
metrics.plot_roc_curve(linear_kernel_svm_clf,x,y)
plt.show()