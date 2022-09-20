import numpy as np
import math


def f1(x):
    y = 1 / (1 + x ** 2)
    return y

def range1(num):
    a = np.array(np.linspace(0, math.pi / 4, num))
    return a

def height1(array):
    h1 = np.zeros(len(array))
    for i in range(len(array)):
        h1[i] = f1(array[i])
    return h1

def f2(x):
    y = 1 / (x * math.sqrt(1 + x))
    return y

def range2(num):
    a = np.array(np.linspace(1, 2, num))
    return a

def height2(array):
    h2 = np.zeros(len(array))
    for i in range(len(array)):
        h2[i] = f2(array[i])
    return h2

def tritangular1(h):
    Area = 0
    for i in range(len(h)):
        Area += 0.5 * (h[i] + h[i + 1]) * (math.pi / (4 * (len(h) - 1)))
        if i + 1 == len(h) - 1:
            break
    return Area

def simpson_even1(h):
    Area = 0
    i =0
    while i <= len(h):
        Area += 2 * (math.pi / (4 * (len(h) - 1))) * 1 / 6 * (h[i] + 4 * h[i + 1] + h[i + 2])
        if i+2  == len(h) - 1:
            break
        i+= 2
    return Area

def simpson_odd1(h):
    Area = 0
    i = 0
    while i <= len(h):
        Area += 2 * (math.pi / (4 * (len(h) - 1))) * 1 / 6 * (h[i] + 4 * h[i + 1] + h[i + 2])
        if i+3 == len(h)-1:
            a = 3/8* (math.pi/(4*(len(h)-1)))*(h[i]+3*h[i+1]+3*h[i+2]+h[i+3])
            Area = Area + a
            break
        i+=2
    return Area

def tritangular2(h):
    Area = 0
    i = 0
    while i <= len(h):
        Area += 0.5 * (h[i] + h[i + 1]) * (1/ (len(h) - 1))
        if i + 1 == len(h) - 1:
            break
        else :
            i +=1
    return Area

def simpson_even2(h):
    Area = 0
    i =0
    while i <= len(h):
        Area += 1/3 * (1/(len(h) - 1)) * (h[i] + 4 * h[i + 1] + h[i + 2])
        if i+2  == len(h) - 1:
            break
        else:
            i+= 2
    return Area

def simpson_odd2(h):
    Area = 0
    i = 0
    while i <= len(h):
        Area += 1/3 * (1/(len(h) - 1)) *(h[i] + 4 * h[i + 1] + h[i + 2])
        if i+3 == len(h)-1:
            a = 3/8* (1/(len(h)-1))*(h[i]+3*h[i+1]+3*h[i+2]+h[i+3])
            Area = Area + a
            break
        else:
            i+=2
    return Area
"""num - even simpson_odd()
num - odd simpson_even()"""
import time
start = time.time()

for num in range(6,100):
    h = height1(range1(num))
    Area = tritangular1(h)
    if np.round(Area,5) == 0.66577:
        print("A- 사다리꼴 계산","등분:",num, "넓이:",Area, np.round(Area,5))
        break

end = time.time()
print(f"{end - start:.5f} sec")

start = time.time()
for n in range(6,100):
    h = height1(range1(n))
    if n%2==0:
        Area = simpson_odd1(h)
    else:
        Area = simpson_even1(h)
    if np.round(Area, 5) == 0.66577:
        print("A- simpson 계산","등분:",n, "넓이:",Area, np.round(Area, 5))
        break
end = time.time()
interval = (end- start)* 2
print(f"{interval:.8f} *0.5 sec")

start = time.time()
for a in range(3,300):
    h = height2(range2(a))
    Area2 = tritangular2(h)
    if np.round(Area2,6) == 0.44579:
        print("B- 사다리꼴 계산","등분:",a,"넓이:", Area2, np.round(Area2,6))
        break
end = time.time()
print(f"{end - start:.5f} sec")

start = time.time()
for b in range(3,500):
    h = height2(range2(b))
    if b%2==0:
        Area = simpson_odd2(h)
    else:
        Area = simpson_even2(h)
    if np.round(Area, 6) == 0.44579:
        print("B- simpson 계산","등분:",b,"넓이:", Area, np.round(Area, 6))
        break
end = time.time()
print(f"{end - start:.5f} sec")

"""Comment
    사다리꼴로 계산을 진행하였들시 각 소구간에 존재하는 오차가 큰것을 확인할수 있다.
    효과적인 계산을 위해서는 simpson 을 사용한는것이 매우 효과적임을 알수 있다."""