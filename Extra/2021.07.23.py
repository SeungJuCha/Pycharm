#3차 배열
import numpy as np
a=np.array([0,1,2,3])
b=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
c= np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])

print(c[1,:]) #c[1,:,:] 과 같음
print(c[1,:,:])   #1차배열,2차배열,3차배열,.....

print(b[[0,2],:])
print(b[[2,0]])
print(b[0][:])

print(b[[0,2],[0,1]]) #0과2의 위치에서 0과 1 즉 0,0/ 2,1출력

b_5=b>5
print(b_5)
print(b[b_5])

print(b[:,[False, True, False, True]]) #b의 True값만 출력됨

d=np.ones_like(b)
print(b+d)

print(b==1)  #broadcasting 자동으로 배열이 복사가 되서 조건에 맞게 수정됨

e=np.zeros((2,3,4))
print(e + b)

print(np.power(b,2)) #b원소의 제곱으로 변환
k1=b.copy()

for i in range(len(k1)):
    for j in range((len(k1[i]))):
        k1[i][j]=k1[i][j]**2
print(k1)

k=np.zeros_like(b)
for i, value in enumerate(b):
    k[i]=value**2
print(k)

print(np.exp(b)) #지수함수

print(np.sum(b,axis=1)) #행방향 합 axis=0이면 열방향 합
print(np.sum(b))

#shape변환 --배열을 나누거나 합치기, 모양 바꾸기, 차원추가
    #split
a=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
print(np.split(a,2),'\n') #기본은 열방향을 자른다
print(np.split(a,(1,3))) #1개 빠지고 2개 빠지고 나머지 범위 1<=, <3
split1, split2 = np.hsplit(a,2)  #기본적으로 열방향으로 고정 (행방향 vsplit)
print("split1 = \n",split1)      #vsplit- vertical hsplit-horizontal
print('split2 = \n',split2)

    #stack!!! 데이터를 합친다는건 쌓는다는것
stack0=np.stack((a,a)) #4by4 +4by4 =2x4x4
print(stack0.shape)
print(stack0)

stack1=np.stack((a,a),axis=1) #4x2x4
print(stack1.shape)
print(stack1)
#차원수를 동일시 하고 그냥 내부에서 합치는것
print(np.vstack((a,a)))
print(np.hstack((a,a)))

    #reshape/resize
b1=a.reshape(2,8)
print('a shape :', a.shape)
print('a reshape :',b1.shape)

b2=a.reshape(8,-1) #음수가 있을시 앞에 양수를 기준으로 말이 되게 바꿈
print('a reshape(-1):',b2.shape)
# reshape를 할때는 사이즈를 계산ㅇ해서 가능해야한다 곱셈
# 사이즈를 감이 안잡히면 약수 하나에 음수 집어넣어서 그냥 맞춰라

#용량이 적을때
a.resize(3,5,refcheck=False) #3x5로 사이즈 축소후 남은거 소멸..
#그냥 하면 오류, refcheck= false로 쓰는가
#ValueError: cannot resize an array that references or is referenced
#by another array in this way.
#Use the np.resize function or refcheck=False 해결법
#아니면
b=np.resize(a,(3,5)) # 대용량일때
print(a)
print(b)

print('-'*40)
#faltten : 복사 / ravel : 참조(view)  array --> 1차로 바꾼다
fla= a.flatten()
fla[0] = 100
print(a)
print(fla)

ravel = a.ravel()
ravel[0]=100
print(ravel)

#transpose 순서를 뒤바꾼다
d= a.transpose() #a=3x5 d=5x3
print(d)
e=a.T #마찬가지 방법
print(e)
print('-'*40)
#squeeze는 차원 축소 사이즈가 1인것만 제거가능
b=np.zeros((1,2,3,4))
print(b.shape)


c=b.transpose(1,0,3,2)
print(c.shape)
print(np.squeeze(c).shape)
print(np.squeeze(c, axis=1).shape)

#newaxis
a=np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
print(a[:].shape)
print(a[np.newaxis,:,:].shape)
print( a[:,np.newaxis,:].shape)
print(a[:,:,np.newaxis].shape)

print('-'*40)
x= np.arange(2*3*4)
print(x)
x=x.reshape(2,3,4)
print('-'*40)
print(x)
print(x[0,:,:])
print(x[:,0,:])
print(x[:,:,0])
print(x.ravel())

print('-'*40)
import matplotlib.pyplot as plt
# x= np.array([1,2,3,4])
# plt.plot(x)
# plt.show()

# plt.plot(x,x**2)
x=np.arange(0,5,0.2)  #색과 모양 지정가능
plt.plot(x,x*2,'ro') #red 동그라미
plt.plot(x,x**2,'b^') #blue 삼각형
plt.plot(x,x**3,'gs') #green 사각형
plt.show()

# x= np.array([1,2,3,4])
# y=np.power(x,2)
# plt.bar(x,y)  #세로 막대
# plt.barh(x,y) #가로형 막대

#점찍는것
# np.random.seed(0) #난수 생성
# x=np.random.rand(50) #50개의 랜덤값
# y=(x*10)+np.random.rand(50)
# plt.scatter(x,y)
# plt.show()

#boxplot
# np.random.seed(0)
# spread = np.random.rand(50)*100
# center = np.ones(25)*25
# flier_high = np.random.rand(10)*100 +100
# flier_low = np.random.rand(10)*-100
# data = np.concatenate((spread,center,flier_high,flier_low))
# plt.boxplot(data)

#piechart

import matplotlib.font_manager as fm # 한글인식을 위해서 집어넣음
for font in fm.fontManager.ttflist:
    print(font.name)

fm.rcParams['font.family']='Malgun Gothic' # 폰트 설정
data= np.array([1,2,3,4])
labels = ['철수','짱구','훈이','맹구']
# labels = ['a','b','c','d']
plt.pie(data,labels=labels)
plt.show()
