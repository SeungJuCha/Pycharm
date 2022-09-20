import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

Diabetes_X, Diabetes_Y = datasets.load_diabetes(return_X_y= True)
Diabetes_X_feature = Diabetes_X[:,np.newaxis,0] #n by n ---> n by 1로 바꿈

Diabetes_X_Train = Diabetes_X_feature[:-20] #처음부터 -21번째까지의 데이터( 선형회귀를 위한 데이터)

Diabetes_X_Test  = Diabetes_X_feature[-20:] #-20부터 끝까지 데이터를 Test (예측에 사용)
Diabetes_Y_Train = Diabetes_Y[:-20]
Diabetes_Y_Test = Diabetes_Y[-20:]  # x_test를 사용했을때 실제 값

Regr = linear_model.LinearRegression()

Regr.fit(Diabetes_X_Train,Diabetes_Y_Train)

Diabetes_Y_Pred = Regr.predict(Diabetes_X_Test) # x_test를 이용한 예측값

# The coefficients
print('Coefficients: \n', Regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Diabetes_Y_Test, Diabetes_Y_Pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Diabetes_Y_Test, Diabetes_Y_Pred))

# Plot outputs
plt.scatter(Diabetes_X_Test, Diabetes_Y_Test,  color='black')
plt.plot(Diabetes_X_Test, Diabetes_Y_Pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
# -------------------------------------------------1차 선형회귀

#polynomial Regression
m = 100
X = 6 * np.random.rand(m, 1) - 3 # rand -> 0과 1 사이의 균일 분포 100 by 1의 array
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1) # randn-> 평균 0 표준편차 1의 표준정규분포

# Plot inputs
plt.scatter(X, y,  color='black')

plt.xlabel('X')
plt.ylabel('y')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2, include_bias = False)  #degree는 차수를 의미
X_poly = poly_features.fit_transform(X)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

# Plot inputs
plt.scatter(X, y,  color='black')
XX = np.linspace(-3, 3, 100).reshape(100,1)  #1by 100 ---> 100by 1로 reshape  # test 값
X_poly_new = poly_features.fit_transform(XX)  #polynomial 을 위한 한단계 fitting 작업
y_poly_new = lin_reg.predict(X_poly_new)

plt.plot(XX, y_poly_new, color='blue', linewidth=2)

plt.xlabel('X')
plt.ylabel('y')
plt.show()

poly_features_degree_three = PolynomialFeatures(degree = 3, include_bias = False)
poly_features_degree_fifty = PolynomialFeatures(degree = 50, include_bias = False)
X_poly_degree_three = poly_features_degree_three.fit_transform(X)
X_poly_degree_fifty = poly_features_degree_fifty.fit_transform(X)

lin_reg_degree_three = linear_model.LinearRegression()
lin_reg_degree_three.fit(X_poly_degree_three, y)
print(lin_reg_degree_three.intercept_, lin_reg_degree_three.coef_)

lin_reg_degree_fifty = linear_model.LinearRegression()
lin_reg_degree_fifty.fit(X_poly_degree_fifty, y)
print(lin_reg_degree_fifty.intercept_, lin_reg_degree_fifty.coef_)

# Plot inputs
plt.scatter(X, y,  color='black')

XX = np.linspace(-3, 3, 100).reshape(100,1)

X_poly_new = poly_features.fit_transform(XX)
y_poly_new = lin_reg.predict(X_poly_new)

X_poly_new_degree_three = poly_features_degree_three.fit_transform(XX)
y_poly_new_degree_three = lin_reg_degree_three.predict(X_poly_new_degree_three)

X_poly_new_degree_fifty = poly_features_degree_fifty.fit_transform(XX)
y_poly_new_degree_fifty = lin_reg_degree_fifty.predict(X_poly_new_degree_fifty)

plt.plot(XX, y_poly_new, color='blue', linewidth=2)
plt.plot(XX, y_poly_new_degree_three, color='green', linewidth=2)
plt.plot(XX, y_poly_new_degree_fifty, color='red', linewidth=2)

plt.rcParams["figure.figsize"] = (70,10)
plt.ylim(-5, 15)
plt.xlabel('X')
plt.ylabel('y')
plt.show()