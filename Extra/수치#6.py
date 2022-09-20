import numpy as np
import pandas as pd
import time

# def fx1 (num):
#   y = 2*np.exp(-3*num)+4
#   return y
#
# def Diff (num):
#   a = 12-3*fx1(num)
#   b = -3* a
#   c = -3*b
#   d = -3*c
#   e = -3*d
#   f = -3*e
#   return (a,b,c,d,e,f)
#
# def Taylor (y,a,b,c,d,e,f):
#   pred = y+0.1*a +1/2*(0.1**2)*b+1/6*b*(0.1**3) + 1/24*c*(0.1**4)
#   +1/120*d*(0.1**5)+1/720*e*(0.1**6)+1/(7*720)*f*(0.1**7)
#   return pred
#
# def fx2 (num):
#   y = -1*np.exp(-1*num)+2*num
#   return y
#
# def Diff_2(num):
#   a = 2+2*num - fx2(num)
#   b = 2-a
#   c = -1*b
#   d = -1*c
#   e = -1*d
#   f = -1*e
#   return(a,b,c,d,e,f)
#
#
# """Problem 1-a"""
# num = np.array(np.linspace(0, 2, 21))
#
# start = time.time()
# Actual = []
# Prediction = []
# for i in range(len(num)):
#   real = fx1(num[i])
#   a, b, c, d, e, f = Diff(num[i])
#   pred = Taylor(real, a, b, c, d, e, f)
#
#   Actual.append(real)
#   Prediction.append(pred)
#
# Actual.pop(0)
# Prediction.pop(-1)
# end = time.time()
# print(str(end - start) + "sec")
#
# import matplotlib.pyplot as plt
#
# x = list(num)
# x.pop(0)
# a = np.linspace(0, 2, 200)
# y = 2 * np.exp(-3 * a) + 4
# z = Prediction
# plt.xlim(0, 2)
# plt.plot(a, y, color='red')
# plt.scatter(x, z)
# plt.show()
#
# num = list(num)
# num.pop(0)
# data = pd.DataFrame({'x': num, 'Actual': Actual, 'Prediction': Prediction})
# print(data)
#
# """Problem 1-b"""
# start = time.time()
# num = np.array(np.linspace(0, 2, 21))
#
# Actual = []
# Prediction = []
# for i in range(len(num)):
#   real = fx2(num[i])
#   a, b, c, d, e, f = Diff_2(num[i])
#   pred = Taylor(real, a, b, c, d, e, f)
#
#   Actual.append(real)
#   Prediction.append(pred)
#
# Actual.pop(0)
# Prediction.pop(-1)
# end = time.time()
# print(str(end - start) + "sec")
#
# import matplotlib.pyplot as plt
#
# x = list(num)
# x.pop(0)
# a = np.linspace(0, 2, 200)
# y = -1 * np.exp(-1 * a) + 2 * a
# z = Prediction
# plt.xlim(0, 2)
# plt.plot(a, y, color='red')
# plt.scatter(x, z)
# plt.show()
#
# num = list(num)
# num.pop(0)
# data = pd.DataFrame({'x': num, 'Actual': Actual, 'Prediction': Prediction})
# print(data)
# #
# """Problem 2-a"""
# def fx3 (num):
#   y = 1/5*np.exp(-3*num)*(np.exp(5*num)+10*np.exp(3*num)-11)
#   return y
# num = np.array(np.linspace(0,2,21))
# num2 = np.array(np.linspace(0,2,201))
#
# Actual = fx3(num)
# Actual = list(Actual)
# Actual.pop(0)
# num = list(num).pop(0)
# h = 0.1
# t = 0
# Y = 0
# y = 0
# Pred_plus = []
# Pred_minus = []
# for i in range(20):
#   Y = (1+3*h)*Y + h*(np.exp(2*t)+6)
#   y = (y+h*(np.exp(2*t)+6))/(1+3*h)
#   t += h
#   Pred_plus.append(Y)
#   Pred_minus.append(y)
#
# error_pl = abs(np.array(Actual)-np.array(Pred_plus))/np.array(Actual)*100
# error_mi = abs(np.array(Actual)-np.array(Pred_minus))/np.array(Actual)*100
# data = pd.DataFrame({'x':num,'Actual': Actual, 'Prediction+':Pred_plus,
#                      'Prediciton-':Pred_minus,'Error+ %':error_pl,
#                      'Error- %':error_mi})
# print(data)
#
# Actual2 = fx3(num2)
# Actual2 = list(Actual2)
# Actual2.pop(0)
# num2 = list(num2).pop(0)
# h = 0.01
# t = 0
# Y = 0
# y = 0
# Pred_plus = []
# Pred_minus = []
# for i in range(200):
#   Y = (1+3*h)*Y + h*(np.exp(2*t)+6)
#   y = (y+h*(np.exp(2*t)+6))/(1+3*h)
#   t += h
#   Pred_plus.append(Y)
#   Pred_minus.append(y)
#
# error_pl = abs(np.array(Actual2)-np.array(Pred_plus))/np.array(Actual2)*100
# error_mi = abs(np.array(Actual2)-np.array(Pred_minus))/np.array(Actual2)*100
# data = pd.DataFrame({'x':num2,'Actual': Actual2, 'Prediction+':Pred_plus,
#                      'Prediciton-':Pred_minus,'Error+ %':error_pl,
#                      'Error- %':error_mi})
# print(data.tail(30))
#
# """Problem 2-b"""
# def fx4 (num):
#   y = 65/32*np.exp(4*num)+num**2/4-num/8-1/32
#   return y
#
# num = np.array(np.linspace(0,2,21))
# Actual = fx4(num)
# Actual = list(Actual)
# Actual.pop(0)
# num = list(num).pop(0)
# h = 0.1
# t = 0
# Y = 2
# y = 2
# Pred_plus = []
# Pred_minus = []
# for i in range(20):
#   Y = (1+4*h)*Y +(t-t**2)*h
#   y = y/(1-4*h) + (h/(1-4*h))*(t-t**2)
#   t += h
#   Pred_plus.append(Y)
#   Pred_minus.append(y)
#
# error_pl = abs(np.array(Actual)-np.array(Pred_plus))/np.array(Actual)*100
# error_mi = abs(np.array(Actual)-np.array(Pred_minus))/np.array(Actual)*100
# data = pd.DataFrame({'x':num,'Actual': Actual, 'Prediction+':Pred_plus,
#                      'Prediciton-':Pred_minus,'Error+ %':error_pl,
#                      'Error- %':error_mi})
# print(data)
#
#
# num2 = np.array(np.linspace(0,2,201))
# Actual2 = fx4(num2)
# Actual2 = list(Actual2)
# Actual2.pop(0)
# num2 =list(num2).pop(0)
# h = 0.01
# t = 0
# Y = 2
# y = 2
# Pred_plus = []
# Pred_minus = []
# for i in range(200):
#   Y = (1+4*h)*Y +(t-t**2)*h
#   y = y/(1-4*h) + (h/(1-4*h))*(t-t**2)
#   t += h
#   Pred_plus.append(Y)
#   Pred_minus.append(y)
#
# error_pl = abs(np.array(Actual2)-np.array(Pred_plus))/np.array(Actual2)*100
# error_mi = abs(np.array(Actual2)-np.array(Pred_minus))/np.array(Actual2)*100
# data = pd.DataFrame({'x':num2,'Actual': Actual2, 'Prediction+':Pred_plus,
#                      'Prediciton-':Pred_minus,'Error+ %':error_pl,
#                      'Error- %':error_mi})
# print(data.head(30))
# #
# #
# """Problem 3"""
# def Real (t):
#   x = 5*np.exp(-t)+3*np.exp(4*t)
#   y = 5 * np.exp(-t) - 2 * np.exp(4 * t)
#   return  (x,y)
#
#
# def DIF (x,y):
#   X_D = 2*x-3*y
#   Y_D = y-2*x
#   return (X_D,Y_D)
#
# def R_K_4(x0,y0):
#   X_k1,Y_k1 = DIF(x0,y0)
#   x1,y1 = (x0+0.05*X_k1, y0 +0.05*Y_k1)
#   X_k2,Y_k2 = DIF(x1,y1)
#   x2,y2 = (x0 +0.05*X_k2, y0 + 0.05*Y_k2)
#   X_k3, Y_k3 = DIF(x2,y2)
#   x3, y3 = (x0 + 0.05 * X_k3, y0 + 0.05 * Y_k3)
#   X_k4, Y_k4 = DIF(x3, y3)
#
#   x_pred =x0+(0.1/6)*(X_k1+2*X_k2+2*X_k3+X_k4)
#   y_pred = y0 + (0.1 / 6) * (Y_k1 + 2 * Y_k2 + 2 * Y_k3 + Y_k4)
#   return (x_pred,y_pred)
#
def error(Act,Pred):

  error = []
  for i in range(len(Act)):
    e = abs((Act[i+1]-Pred[i])/Act[i+1])*100
    error.append(e)
    if i+1 == len(Act)-1:
      break
  return error
#
# start = time.time()
# t = np.linspace(0,10,101)
# Actualx,Actualy = Real(t)
#
# Predictx = []
# Predicty = []
# x0,y0 = (8,3)
# for i in range(len(t)):
#   x_p,y_p = R_K_4(x0,y0)
#   Predictx.append(x_p)
#   Predicty.append(y_p)
#   x0,y0 = (x_p,y_p)
#
# # Predictx = np.array(Predictx)
# # Predicty = np.array(Predicty)
#
# Error_x = error(Actualx,Predictx)
# Error_y = error(Actualy,Predicty)
# end = time.time()
# print(str(end-start)+'sec')
#
# t = list(t)
# t.append(10.1)
# Actualx = list(Actualx)
# Actualx.append('-')
# Actualy = list(Actualy)
# Actualy.append('-')
# Predictx = list(Predictx)
# Predictx.insert(0, '-')
# Predicty = list(Predicty)
# Predicty.insert(0, '-')
# Error_x.append('-')
# Error_x.insert(0,'-')
# Error_y.append('-')
# Error_y.insert(0,'-')
#
# data = pd.DataFrame({'t':t,'Actual_x':Actualx , 'Prediction_x':Predictx,
#                      'Actual_y':Actualy , 'Prediction_y':Predicty,
#                      'Error_x %':Error_x,'Error_y %': Error_y})
# print(data)

"""Problem 4"""
import sympy as sy
def Integrate (t):
  x = sy.symbols('x')
  I = 5*sy.Integral(sy.cos(2*(t-x))*(1/27*(24+120*x+30*sy.cos(3*x)+50*sy.sin(3*x))),(x,0,t))
  NUMB = I.evalf()
  return NUMB

def RealFX(x):
  y = 1/27 * (24 + 120*x + 30*np.cos(3 * x) + 50 * np.sin(3 * x))
  return y

def Dif_Prob4 (t):
  Y_D = 10-Integrate(t)
  return Y_D


def R_K_4_Prob4 (t0,y0):
  Y_k1 = Dif_Prob4(t0)
  t1,y1 = (t0+ 0.05,y0 + 0.05 * Y_k1)
  Y_k2 = Dif_Prob4(t1)
  t2,y2 = (t0+ 0.05,y0 + 0.05 * Y_k2)
  Y_k3 = Dif_Prob4(t2)
  t3,y3 = (t0+ 0.1, y0 + 0.05 * Y_k3)
  Y_k4 = Dif_Prob4(t3)

  y_pred = y0 + (0.1 / 6) * (Y_k1 + 2 * Y_k2 + 2 * Y_k3 + Y_k4)
  return y_pred


start = time.time()
t = np.linspace(0,10,101)
Actualy_Prob4 = RealFX(t)

Predicty_Prob4 = []
y0 = 2
for i in range(3):
  y_p = R_K_4_Prob4(t[i],y0)
  Predicty_Prob4.append(y_p)
  y0 = y_p


Predicty_Prob4.insert(0,2)
# [2, 2.94207936656818, 3.74045396134088, 4.36350825808219,4.74562002097910]
"""RUnge Katta 로 4개의 y값 이제 이것으로 기울기 필요"""
Yprime_Pred_list = []
for i in range(4):
  Y_Prime = Dif_Prob4(t[i])
  Yprime_Pred_list.append(Y_Prime)

# [10, 8.76680202849335, 7.14750072707032, 5.28674345830097,a]
for i in range(len(t)):
  y_Predictor = Predicty_Prob4[i+3]+0.1/24*(55*Yprime_Pred_list[i+3]-59*Yprime_Pred_list[i+2]+37*Yprime_Pred_list[i+1]
                                       -9*Yprime_Pred_list[i])

  yprime_Predictor = Dif_Prob4(y_Predictor)

  y_Corrector = Predicty_Prob4[i+3]+0.1/24*(9*yprime_Predictor+19*Yprime_Pred_list[i+3]
                                        -5*Yprime_Pred_list[i+2]+Yprime_Pred_list[i+1])

  y_Predictor_Modified = y_Predictor+251/270*(y_Corrector-y_Predictor)

  yprime_Modified = Dif_Prob4(y_Predictor_Modified)

  y_Corrector_new = Predicty_Prob4[i+3]+0.1/24*(9*yprime_Modified+19*Yprime_Pred_list[i+3]
                                        -5*Yprime_Pred_list[i+2]+Yprime_Pred_list[i+1])

  y_Corrector_new_Modified = y_Corrector_new-19/270*(y_Corrector-y_Predictor)

  yprime_Corrector_Modified = Dif_Prob4(y_Corrector_new_Modified)
  Predicty_Prob4.append(y_Corrector_new_Modified)
  Yprime_Pred_list.append(yprime_Corrector_Modified)
  if i+3 == len(t)-2:
    break
end = time.time()
print(str(end-start)+'sec')


error_4 = error(Actualy_Prob4,Predicty_Prob4)
error_4.insert(0,0)

data = pd.DataFrame({'x': t, 'Actual': Actualy_Prob4, 'Prediction':Predicty_Prob4, 'Error': error_4})
print(data.head(20))

"""comment of Prob 4
  1번의 mop up 계산을 하는데 중간에 적분까지 섞여있기에 7초라는 시간이 소요되었다. 즉 이론상 0~10 까지 0.1까지의 간격으로 
  계산을 진행하게 되면 """