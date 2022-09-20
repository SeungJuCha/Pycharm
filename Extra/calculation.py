"""problem #4

    given : math.e**(math.sin(x))를 0~pi/2 까지 적분
    find : 정적분의 정의를 사용하여 유효숫자 3자리 까지 정확하게 구해라
    schematic:"""
import math
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

x_axis = np.linspace(0, math.pi/2, 100)
y_axis = np.exp(np.sin(x_axis))
plt.plot(x_axis,y_axis, color = 'Red')

plt.xlabel('x_axis')
plt.ylabel('y_axis')
plt.show()
"""assumption: 정적분의 정의는 그래프의 내부를 작은 사각형으로 쪼꺠 dA=dxdy를 구한뒤
    그것들을 전부 더하는 과정이다.
    근사상대오차 판별법인 Scarborough 판정법을 따른다"""

"""Analysis:"""
"""Scarborough 판정법을 통한 유효숫자 3자리
    상대오차 판별"""
n = 3
Scarborough_error = 0.5*10**(1-n)

"""참값"""
x = sym.Symbol('x')
F_x = sym.integrate(sym.exp(sym.sin(x)),(x,0,sym.pi/2))
Xt = 3.10437901785556

"""적분 범위를 divide 하는 수로 나눠 array로 받는 함수
    사각형의 높이인 함수값 구할시 array의 index를 사용하기 위함"""
def Range (div_number):
    x = np.linspace(0, np.pi/2.0, div_number)
    x_array = np.array(x)
    return x_array

""" Fx 계산값"""
def base_function (x):
    Fx = math.e**(math.sin(x))
    return Fx
""" for문을 통해 dA값의 전체합을 구하는 구분구적법
    내부 적분을 사용하며,i= 1부터 시작하고 마지막에 i = 1000일 떄 발생하는
    오류를 방지하기 위해 if 문을 사용한 탈출
    유효숫자 3자리 까지 사용을 위한 근사값 출력"""
def mensuration_of_division (array):
    A_total = 0
    for i in range(len(array)):

        if i + 1 < len(array):

            dy = base_function(array[i+1])
            dA = dy*(array[i+1]-array[i])
            A_total += dA

        else:
            break
    A_total_a = round(A_total, 2)
    return A_total_a


"""백분율 상대오차 et와 근사상대오차 ea를 계산"""
def abs_error (Xt,div_number):
    Xa = mensuration_of_division(Range(div_number))
    Xa1 = mensuration_of_division((Range(div_number+1)))
    et = abs((Xt-Xa)/Xt)*100.0
    es = abs((Xa1-Xa)/Xa1)*100.0 # 근사 백분율
    error_array = np.array([et,es])
    return error_array

"""계산 TIME INTERVAL """
import time
start = time.time()
math.factorial(100000)


"""무한 루프: 근사 상대오차가 Scarborough 값보다 작거나 같은경우 loop를 멈추고 그때의 값들을 출력"""
n_lp = 0
while n_lp <300:
    n_lp = n_lp + 1
    error_array = abs_error(Xt,n_lp)


    if error_array[1] <=  Scarborough_error :
        print(n_lp)
        print('정적분의 값 {}'.format(mensuration_of_division(Range(n_lp))))
        print('error {}'.format(abs_error(Xt, n_lp)))
        break
    else:
        continue

end = time.time()
print(f"{end - start:.5f} sec")

"""Comment: 참값의 유효숫자 3자리까지의 값은 3.10 이었으나 출력된 값은 3.19였다. 이것의 이유는
    루프를 돌면서 근사상대오차의 값들을 계산시 3.10에 가까워 지기도 전에 이미 근사값들의 오차가 
    Scarborough 값 내부로 들어왔다는 것을 알수 있다. 그렇기에 만약 참값을 모른다고 가정했을시 본인은
    3.19를 이것의 적분값으로 판단할수도 있다. 즉 Scarborough 판정만으로는 무엇인가 부족한 점들이 존재한다고 
    생각한다."""
