from matplotlib import ticker

y1=1.1235
y2=3.178
y3=y1*y2
y4=1/3
print("y1 = {0:.4f}, y2 = {1:.1f}, y3 = {2:.3f}, y4 = {3:.5f}".format(y1, y2, y3, y4))  #소수점 나타내기

print(isinstance(1.2, int)) #1.2가 int 인가
print(complex(2,5).conjugate()) #복소수의 부호반대

list = [1,2,3,4,5]
list2 = [1,2,3,4]
"""pop() index 넘버를 뺀다
   append() 그 값을 마지막에 붙인다
   extend() 안의 변수에 해당하는것을 전체다 붙인다
   unpacking 방법 (*list)
   두개를 묶는법 zip"""

tuple=(1,2,'kr','1.2')
#list 와 tuple의 다른점 changeable 여부

setA = set(['사과', '펜', '펜', '파인애플']) # identical element is not included
"""add, remove 사용 프린트시 순서 섞임
   intersection() 겹치는거 확인"""

dict = {1:'Usa',10:'Korea',3:'Japan'} #key & value의 조합
print(dict[10])
dict[10] = 'England'
print(dict.values())
"""key---> value는 불러오기 가능 반대는 불가능
   .keys(), .values() 등으로 어떤 값이 있는지 알수 있음
   value값 변환 가능
   unpacking 방법 (**dict)"""
num = [2, 4, 6, 8, 10]
for i, n in enumerate(num):
    """enumerate는 index값과 value를 사용하는것"""
    print(f'{i} and {n}')
    num[i] = n * 2
print(num)

# map from letter grade to point value: "dictionary object"
points = {'A+':4.5, 'A':4.0, 'B+':3.5, 'B':3.0,
          'C+':2.5, 'C':2.0, 'D+':1.5,'D':1.0, 'F':0.0}

# initialize variables
num_courses = 0
total_points = 0
done = False

# while loop
# while not done:
#   grade = input()                          # read line from user
#   if grade == '':                          # empty line was entered (blank)
#     done = True
#   elif grade not in points:                # unrecognized grade entered
#     print("Unknown grade '{0}' being ignored".format(grade))
#   else:
#     num_courses += 1
#     total_points += points[grade]
# if num_courses > 0:                        # avoid division by zero
#   print('Your GPA is {0:.3}'.format(total_points / num_courses))

"""List comprehension"""
list = [(i,j) for i in range(2) for j in range(3)]
print(list)

import numpy as np
"""np.zeros((m,n))---mbyn 을 0으로 채움
   np.ones((m,n))---1로 채움
   np.eye(m)---mby m의 단위 행렬"""
na_arange = np.arange(10,25,7) # 10 ~24를 7단위로 건너뛴다
na_linspace = np.linspace(10,25,3) #10~25를 3등분
na_random = np.random.random((2,2)) #0~1사의의 랜덤값
Array = np.array([[1,2,3],[2,2,5],[4,7,3],[2,8,0]])

print(Array.reshape(3,-1))
Array2 = np.hstack((Array,Array))
Array3 = np.vstack((Array,Array))
print(Array2)
print(Array3)
# 사이즈를 모를때는 resize도 사용이 가능하나 데이터가 남을경우 refcheck = False로 소멸가능

import matplotlib.pyplot as plt
Z = np.random.uniform(0, 1, (8,8))

#imshow == color box  contour == 등고선 느낌
plt.contour(Z)
plt.colorbar(label= 'color')
plt.show()

def f(x, y):
    return (x * x) - (y * y) - 1
    #Hyperbola

n = 200
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
XX, YY = np.meshgrid(x, y) #좌표 묶기
ZZ = f(XX, YY)

plt.title("Contour plots")
plt.contourf(XX, YY, ZZ)
plt.contour(XX, YY, ZZ, colors='black')
plt.show()

Z = [10,20,30,40]
plt.pie(Z, labels=['blue','orange','green','red'], autopct='%1.1f%%')
plt.show()

Z = np.random.normal(0, 1, 1000)
Z2 = np.random.normal(5, 1, 1000)
fig, ax = plt.subplots()
ax.hist(Z, color = "red", alpha=0.1)  # alpha 투명도
ax.set_title("Histograms")
ax.hist(Z2, color = "green", alpha=1)
plt.show()

X = np.linspace(0, 4*np.pi, 100)
Y1 = np.sin(X)
Y2 = np.cos(X)
fig, ax = plt.subplots()
ax.plot(X, Y1, color="green", linewidth="3", label="sin")
ax.plot(X, Y2, color="red", linewidth="3", label="cos")
# plt.legend()  ---라벨 표시
ax.annotate("Something special!",(X[30],Y1[30]),(X[30],Y1[70]),ha='center', va='center',
            arrowprops = {"arrowstyle": "->", "color" : "black", "linewidth":"3"})
plt.show()
X = np.linspace(0, 4*np.pi, 100)
Y1 = np.sin(X)
Y2 = np.cos(X)
Y3 = np.exp(X)
Y4 = X * X

plt.style.use("default")

import matplotlib.ticker
fig, axs = plt.subplots(2,2, figsize=(10,8))
axs[0, 0].plot(X, Y1, color="green", linewidth="3", label="sin")
axs[0, 0].legend(loc="upper left")
axs[0, 0].xaxis.set_major_locator(ticker.NullLocator())
axs[0, 0].set_ylim(-1,1)
axs[0, 0].yaxis.set_major_locator(ticker.LinearLocator(numticks=3))
axs[0, 1].plot(X, Y2, color="red", linewidth="3", label="cos")
axs[0, 1].legend(loc="upper left")
axs[1, 0].plot(X, Y3, color="blue", linewidth="3", label="exp(X)")
axs[1, 0].legend(loc="upper left")
axs[1, 0].set_yscale("log")
axs[1, 1].plot(X, Y4, color="orange", linewidth="3", label="X^2")
axs[1, 1].legend(loc="upper left")
plt.tight_layout()
# plt.show()

width = 0.3

fig, ax = plt.subplots(figsize=(10,8))
ax.bar(X-width/2, Y1,width)

plt.rcParams['hatch.color'] = 'red'
plt.rcParams['hatch.linewidth'] = 8

ax.bar(X+width/2, Y2,width, hatch="/")
# plt.show()

#3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

# plt.show()
import numpy as np
import pandas as pd
data = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
        'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(data)

df =df.rename({'one':'three','two':'four'},axis=1)
print(df)


from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test =\
        train_test_split(diabetes_X,diabetes_y, test_size= 0.33,random_state= 42)

# diabetes_X_feautre = diabetes_X.reshape(-1,1)
# diabetes_X_feautre = diabetes_X.reshape(-1,1)
# print(len(diabetes_y))
# print(len(diabetes_X_feautre))
# diabetes_X_train = diabetes_X_feautre[:-20]
# diabetes_X_test = diabetes_X_feautre[-20:]
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]
# print(len(diabetes_X_train))
# print(len(diabetes_y_train))

regr = LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test,y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, y_pred))
# #
from sklearn.linear_model import LinearRegression
m = 100
X = 6 * np.random.rand(m, 1) - 3 # rand -> 0과 1 사이의 균일 분포
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)# randn-> 평균 0 표준편차 1의 표준정규분포

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

XX = np.linspace(-3, 3, 100).reshape(100,1)
X_poly_new = poly_features.fit_transform(XX)
y_poly_new = lin_reg.predict(X_poly_new)

poly_features_degree_three = PolynomialFeatures(degree = 3, include_bias = False)
poly_features_degree_fifty = PolynomialFeatures(degree = 50, include_bias = False)
X_poly_degree_three = poly_features_degree_three.fit_transform(X)
X_poly_degree_fifty = poly_features_degree_fifty.fit_transform(X)

lin_reg_degree_three = LinearRegression()
lin_reg_degree_three.fit(X_poly_degree_three, y)
print(lin_reg_degree_three.intercept_, lin_reg_degree_three.coef_)

lin_reg_degree_fifty =LinearRegression()
lin_reg_degree_fifty.fit(X_poly_degree_fifty, y)
print(lin_reg_degree_fifty.intercept_, lin_reg_degree_fifty.coef_)