question =['name','quest','favorite color']
answers = ['lancelot','the holy grail','blue']
for q,a in zip(question,answers): #묶기 dic형태
    #print('What is your {0}? It is {1}'.format(q,a))
    print('What is your {}? It is {}'.format(q, a))

for i in reversed(range(1,10,2)): #반대로 프린팅
    print(i)

basket = {'apple','orange','apple','pear','orange','banana'}

for i in set(sorted(basket)): #바뀜
    print(i)
    pass
for f in sorted(set(basket)): # 바뀌지 않음
    print(f)

import math
raw_day=[56.2,float('NaN'),51.7,55.3,52.5,float('NaN'),47.8]
filtered_data=[]
for value in raw_day:
    if not math.isnan(value):
        filtered_data.append(value)
print(raw_day)
print(filtered_data)

string1,string2,string3 ='','first','second'
a=string1 or string2 or string3 # 공백이 아닌문자 =문자열의 참
b= string1 and string2 and string3 #string1이 False기에 탈락
print(a)

#if 와 not if 를 and 와 or로 만들수 있다 !!! 튜닝의 기본

a=1
b=2

c=0
a=(a>b) and (c:=a) #처음이 거짓
print(c)

a=1
a=(a>b) or (c:=a) #앞에가 false지만 뒤에가 참이므로
print(c)

a=1
#if-else문
a=((a>b)and (c:=a))or ((a<b) and (c:=b))
print(c)

#v피보나치라는 함수를 끌어다 쓰는법
import Fibonacci

f100= Fibonacci.fib2(100) #파일의 함수명에 값을 대입
print(f100)

import Fibonacci as f #파일을 F로 받는다
f.fib2(1000)

from Fibonacci import fib2 as fibs #인공지능에서 많이 씀
print(fibs(1000)) # from Finonacci import * 는 모든것을 가져옴

import Fibonacci #IMPORT대상이 함수가 아니라 모듈일경우
import importlib #이미 로딩이 된걸 수정해서 다시 로딩할때
importlib.reload(Fibonacci) #를 통해모듈을 다시 로딩

#Numpy
import numpy as np #만약 numpy라는 파일이 존재하면 그것이 더 우선이다!
print(np.__version__)
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array(a+b)
print(a)
print(b)
print(c)

# while True:
#     try:
#         x=int(input('please enter a nuumber'))
#         break
#     except ValueError:
#         print('try again')

class B(Exception): #exception을 상속받은 B라는 class
    pass
class C(B): #B를 받은 C라는 class
    pass
class D(C):
    pass
for cls in [B,C,D]:
    try:
        raise cls()
    except D:
        print("D")  #B는 D보다 훨씬 먼저이기에 안걸림
    except C:
        print("C") # B는 c보다 위에 존재
    except B:
        print("B") #B가 걸리기에 B가 출력

for cls in [B, C, D]:
    try:
        raise cls()  #ㅠB,C,D가 차례로 들어간다
    except B:
        print("B")  # B에서 바로 걸려버림 전체가다
    except C:
        print("C")
    except D:
        print("D")

#exception은 b,c,d를 포함
#b는 c,d를 포함
#c는 d를 포함
import sys
for arg in sys.argv[1:]:
    try:
        f =open(arg,'r')
    except OSError:
        print('cannot open',arg)
    else:
        print(arg,'has',len(f.readlines()),'lines')
    f.close()

try:
    raise Exception('spam','eggs')
except Exception as inst:
    print(type(inst)) # the exception instance
    print(inst.args) # arguments stored in .args
    print(inst) # __str__ allows args to be printed directly,
 # but may be overridden in exception subclasses
    x,y =inst.args # unpack args
    print('x =',x)
    print('y =',y)

def this_fails():
    x=1.0/0
try:
    this_fails()
except ZeroDivisionError as err:
    print('Handling run-time error:',err)

#뒷정리 동작
a=0
try:
    print('try TRY')
    if a==0:
        raise KeyboardInterrupt
    print('End of try')
except KeyboardInterrupt:
    print('Exception OK')
finally:
    print('Goodbye, world!')

def bool_return():
    try:
        return True
    finally:  #return하기전에 항상 실행된다
        return False
print(bool_return())

#지도 학습 = 문제지 &답지가 존재(라벨이 존재) y애 맞는 x를예측
#비지도 학습= 답지가 없는 (라벨이 없음)

import numpy as np
# import time
# n=1000
# m=10000
# x=np.random.rand(n,m)
# w=np.random.rand(n,1)  #n by 1행렬의 랜덤한 값을 가짐
#
# t1= time.time() #시간 계산 하는 라이브러리(현재시간=시작시간)
# z1= np.zeros((1,m)) #1 by m으로 만들어진 0으로만 이루어진 행렬
# for i in range(x.shape[1]): #x의 열
#     for j in range(x.shape[0]):#x의 행
#         z1[0][i]+=w[j]*x[j][i]
# print('반복문 사용코드의 속도:',(time.time()-t1)*1000,'ms')
#
# t2=time.time()
# z=np.dot(w.T,x) #dot 연산
# print('Vectorization 코드의 속도:',(time.time()-t2)*1000,'ms')

array = np.arange(5,10,2) #5~9에서 +2 단위 5,7,9
print(array)

zeroArray = np.zeros(array) #5by7by9 3차원 행렬을 0으로 채운다
print(zeroArray)

oneArray = np.ones(array) #1로 채운다
print(oneArray)

fullArray= np.full((2,3),2) #2by3을 2로 채운다
print(fullArray)

emptyArray = np.empty((2,3)) #2by3을 초기화는 하지 않는다
print(emptyArray) #쓰레기 값 존재

a= np.array([1,2,3,4,5])
b= np.zeros_like(a) #a와 크기가 같은 것에 0으로 채운다
print(b)
c= np.empty_like(a)
print(c) #쓰레기 값으로만 채워진 a와 크기가 같은 행렬

#등간격 갯수 지정
space = np.linspace(0,30,5)  #0부터 30까지를 5개숫자로 등간격표현
print(space) #linesapce(start, end , num) num읙 개수로 등분을 해라

#n차 행령의 n 값 출력
print(oneArray.ndim)

#행렬 형태=크기
print(oneArray.shape)

print(oneArray.size)

print(oneArray.dtype)
#원소의 자료형을 바이트 크기로 출력
print(oneArray.itemsize)
#한 원소에서 다음 원소까지의 거리 (byte)
#5x7x9
#dtype = float64 -> 8byte *8
#5x7x(9) :0 ->1 까지 거리(byte) 8byte
#5x(7) : 8byte *9 -?72
#(5) : 8x9x7-> 504byte 한 차원 사이의 거리
#toal크기 = 5x7x9x8
print(oneArray.strides)

print('------------------------------------------------------------')
a=np.array([0,1,2,3])
b=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
print('B행렬 형태', b.shape)
print('B원소 타입',b.dtype)
print('B원소 거리',b.strides)

print(b[0])
print(b[0][1])
print(b[0,1]) #위말과 같음
print(b[:1])
print(a[1:4:2])
print(a[::-1])

print(b[:,0]) #1째 행 프린트
print(b[0,:]) #1째 열 프린트
print(b[:])

