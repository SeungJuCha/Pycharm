 5import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
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

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0) # logical_xor(X1,X2) X1과 X2중 하나만 true일 경우에만 true return
y_xor = np.where(y_xor, 1, -1) # y_xor가 true이면 1을 false이면 -1을 할당

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sc = StandardScaler()
sc.fit(X_xor)
X_train_std = sc.transform(X_xor)

"""Using polynomial kernel"""
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=10, coef0=1, C=1000))  # default: coef0=0 얼마나 영향을 받을지 차수에
])
poly_kernel_svm_clf.fit(X_train_std,y_xor)
plot_decision_regions(X_train_std, y_xor, poly_kernel_svm_clf)
plt.show()

"""Using Gaussian Kernel"""
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=10, C=1000))
    ])
rbf_kernel_svm_clf.fit(X_train_std,y_xor)
plot_decision_regions(X_train_std, y_xor, rbf_kernel_svm_clf)
plt.show()


"""GridSearchCV"""
# from sklearn.model_selection import GridSearchCV
# estimator = SVC()
# params = {'C':[1,3,5],'kernel':['linear','poly','rbf','sigmoid'],'degree':[3,5,7,10]
#           ,'gamma':[0.5,1,5,10],'coef0':[1,10,100],'random_state':[10,28,42]}
# model = GridSearchCV(estimator = estimator, param_grid = params
#                      ,verbose=1, n_jobs=-1, refit= True)
# model.fit(X_train_std,y_xor)
#
# print('Best Estimator : \n', model.best_estimator_)
# print()
# print('Best params :\n', model.best_params_) #최적의 변수
# print()
# print('Best score: \n',model.best_score_)

from sklearn import  metrics, model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_xor, y_xor, test_size= 0.33, random_state=42)
new_rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=10, C=1000, probability=True))
    ])
new_rbf_kernel_svm_clf.fit(X_train,y_train)

# Confusion Matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = metrics.plot_confusion_matrix(
        new_rbf_kernel_svm_clf, X_test, y_test,
        cmap=plt.cm.Blues, normalize=normalize
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

#ROC cURVE
from sklearn import metrics
metrics.plot_roc_curve(new_rbf_kernel_svm_clf, X_test, y_test)
plt.show()