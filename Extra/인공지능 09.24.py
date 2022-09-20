import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (x * x) - (y * y) - 1
    #Hyperbola

n = 200
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
XX, YY = np.meshgrid(x, y)
ZZ = f(XX, YY)

plt.title("Contour plots")
plt.contourf(XX, YY, ZZ)
plt.contour(XX, YY, ZZ, colors='red')
plt.show()

# n = 100
# x = np.linspace(-10, 10, n)
# y = np.linspace(-10, 10, n)
# XX, YY = np.meshgrid(x, y)
# ZZ = f(XX, YY)
#
# fig, ax = plt.subplots()
#
# ax.set_title("Contour plots")
# subplot = ax.contourf(XX, YY, ZZ)
# ax.contour(XX, YY, ZZ, colors='black')
# ax.set_aspect('equal')
# fig.colorbar(subplot, label='Value')
# fig.show()


X = np.linspace(0, 4*np.pi, 100)
Y1 = np.sin(X)
Y2 = np.cos(X)
fig, ax = plt.subplots()
ax.plot(X, Y1, color="green", linewidth="3", label="sin")
ax.plot(X, Y2, color="red", linewidth="3", label="cos")
ax.set_title("Sine & Cosine")
ax.annotate( 'Special',(X[30],Y1[30]),(X[50],Y1[70]),ha='center', va='center',
             arrowprops = {"arrowstyle": "->", "color" : "black", "linewidth":"3"})
fig.legend(loc="upper left",ncol = 2)

plt.show()

X = np.linspace(0, 4*np.pi, 100)
Y1 = np.sin(X)
Y2 = np.cos(X)
Y3 = np.exp(X)
Y4 = X * X

plt.style.use("grayscale") #모듈 자체 스케일을 gray로 설정(배경)
fig, axs = plt.subplots(2,2,figsize = (12,8))  #2by2짜리 그래프 총 4개를 그린다

axs[0, 0].plot(X, Y1, color="green", linewidth="3", label="sin")
axs[0, 1].plot(X, Y2, color="red", linewidth="3", label="cos")
axs[1, 0].plot(X, Y3, color="blue", linewidth="3", label="exp(X)")
axs[1, 1].plot(X, Y4, color="orange", linewidth="3", label="X^2")
fig.tight_layout() #그래프 크기를 따로 설정하지 않고 전체적으로 고루고루 사용하기위함
fig.legend(loc="upper left",ncol=2)
#따로따로 legend 설정하려면 axs[0,0].legend 형식으로 따로따로 설정가능

plt.show()

#3D
## https://matplotlib.org/3.1.1/gallery/mplot3d/surface3d.html

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, #cmap = color map
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

"""3D 그리는 법
1. 데이터를 만든다
2. surface 그린다
3. z축 설정
4. 선위에 표면을 얹는다"""

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve',color = 'blue')
ax.legend()

plt.show()