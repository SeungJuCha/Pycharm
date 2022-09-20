import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model

File = pd.read_csv('data.csv')
X = File['X']
y = File['y']
# XX =X.reshape((-1,1))------오류 type(X) = Series
X = X.to_numpy()
XX = X.reshape(-1,1)
X_train = XX[:-30]
y_train = y[:-30]
X_test = XX[-30:]
y_test = y[-30:]


lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
print(lin_reg.intercept_,lin_reg.coef_)
print('Coefficient of determination:'
      , r2_score(y_test, y_pred))

#n차 polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
Poly_feature_2 = PolynomialFeatures(degree= 2, include_bias= False)
Poly_feature_3 = PolynomialFeatures(degree = 3, include_bias = False)
Poly_feature_5 = PolynomialFeatures(degree= 5, include_bias= False) #best 값
Poly_feature_10 = PolynomialFeatures(degree=10, include_bias= False)
#
X_train_2 = Poly_feature_2.fit_transform(X_train)
X_train_3 = Poly_feature_3.fit_transform(X_train)
X_train_5 = Poly_feature_5.fit_transform(X_train)
X_train_10 = Poly_feature_10.fit_transform(X_train)

X_test_2 = Poly_feature_2.fit_transform(X_test)
X_test_3 = Poly_feature_3.fit_transform(X_test)
X_test_5 = Poly_feature_5.fit_transform(X_test)
X_test_10 = Poly_feature_10.fit_transform(X_test)

lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_train_2,y_train)
y_pred_2 = lin_reg_2.predict(X_test_2)
print(lin_reg_2.intercept_,lin_reg_2.coef_)
print('Coefficient of determination:'
      , r2_score(y_test, y_pred_2))

lin_reg_3 = linear_model.LinearRegression()
lin_reg_3.fit(X_train_3,y_train)
y_pred_3 = lin_reg_3.predict(X_test_3)
print(lin_reg_3.intercept_,lin_reg_3.coef_)
print('Coefficient of determination:'
      , r2_score(y_test, y_pred_3))

lin_reg_5 = linear_model.LinearRegression()
lin_reg_5.fit(X_train_5,y_train)
y_pred_5 = lin_reg_5.predict(X_test_5)
print(lin_reg_5.intercept_,lin_reg_5.coef_)
print('Coefficient of determination:'
      , r2_score(y_test, y_pred_5))
#
#
lin_reg_10 = linear_model.LinearRegression()
lin_reg_10.fit(X_train_10,y_train)
y_pred_10 = lin_reg_10.predict(X_test_10)
print(lin_reg_10.intercept_,lin_reg_10.coef_)
print('Coefficient of determination:'
      , r2_score(y_test, y_pred_10))


#####plotting
plt.scatter(X_train, y_train,  color='black')
XX = np.linspace(-3, 3, 100).reshape(100,1)

X_poly_new = Poly_feature_2.fit_transform(XX)
y_poly_new = lin_reg_2.predict(X_poly_new)

X_poly_new_degree_three = Poly_feature_3.fit_transform(XX)
y_poly_new_degree_three = lin_reg_3.predict(X_poly_new_degree_three)

X_poly_new_degree_five = Poly_feature_5.fit_transform(XX)
y_poly_new_degree_five = lin_reg_5.predict(X_poly_new_degree_five)

X_poly_new_degree_ten = Poly_feature_10.fit_transform(XX)
y_poly_new_degree_ten = lin_reg_10.predict(X_poly_new_degree_ten)

plt.plot(XX, y_poly_new, color='blue', linewidth=2)
plt.plot(XX, y_poly_new_degree_three, color='green', linewidth=2)
plt.plot(XX, y_poly_new_degree_five, color='red', linewidth=2)
plt.plot(XX, y_poly_new_degree_ten, color='yellow', linewidth=2)

plt.rcParams["figure.figsize"] = (70,10)
plt.ylim(-2, 5)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

#최적 fitting curve