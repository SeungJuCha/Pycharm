print(b'Hello, world')
#문자열을 바이트 배열 객체 생성
#b'Hello, world'
print(bytes(10))
#0이 10개 들어있는 바이트 배열 객체 생성
#b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
print(bytes([10,20,30,40,50]))
#반복 가능 객체로 바이트 배열 객체 생성
#b'\n\x14\x1e(2'
print(bytes(b'Hello'))
#바이트 배열 객체로 바이트 배열 객체 생성
#b'Hello'
print(bytearray(10))
#0이 10개 들어있는 바이트 배열 객체 생성
#bytearray(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
print(bytearray([10,20,30,40,50]))
#반복 가능 객체로 바이트 배열 객체 생성
#bytearray(b'\n\x14\x1e(2')

bytearray(b'Hello')

x=bytearray(b'Hello')
x[0]= ord('a') #ord는 문자의 아스키 코드값을 변환 ->1byte로 변환
print(x)  #->문자열의 슬라이싱[index] 변환이 가능하다
#bytes,str은 수정이 불가하기에...

#파이썬의 문자열 인코딩 방식은 항상 UTF-8
s='한글을 사랑합시다'
b= b'Hello,world!'
#이렇게 하면 ASCII코드 방식으로 인코딩을 한다
#만약 다른 문자가 포함되면 에러발생(한글,기타등등)

#문자열 객체를 바이트 배열 객체로 인코딩 하는법: encode(방식)함수
#방식을 지정해주지않으면 기본적으로 UTF-8사용

b_1=s.encode()
b_2=s.encode('UTF-8')
b_3=s.encode('euc-kr') #ksc5601
b_4=s.encode('ms949')
b_5=s.encode('cp949') # 공통 문자셋이며 ms949, ksc5601과 거의 비슷
#b5=s.encode('ascii')
#ascii' codec can't encode characters in position 0-2: ordinal not in range(128)
#아스키 코드에는 한글이 없다 그래서 encoding 실패
print(b_1,'\n',b_2,'\n',b_3,'\n',b_4,'\n',b_5)

#반대로 배열객체를 문자열로 하는법 decode()
#str로 쓰는 것보다 더 빠름
s1=b_1.decode()
s2=b_2.decode('utf-8')
s3=b_3.decode('euc-kr')
s4=b_4.decode('ms949')
# s5=b_5.decode('ascii')
print(s1,'\n',s2,'\n',s3,'\n',s4)

print('-' *40)

fruits = ['orange','apple','pear','banana','kiwi','apple','banana']
print(fruits.count('apple'))  #count()는 개수를 세라
print(fruits.count('tangerine'))

#바나나를 모두찾아서 해당위치 출력
idx=0
for i in range(fruits.count('banana')):
    idx= fruits.index('banana',idx+1)
    # fruits 리스트에서 banana가 있는 위치를 찾는거 index(값, 스타트점 )
    #index(value, start, end) ,value의 index를 찾는데 start에서 end까지
    print(idx, end=', ')
print()

print(fruits.index('banana',4))
fruits.reverse()
print(fruits)
fruits.append('grape')
print(fruits)
fruits.sort()  #알파벳 순으로나열
print(fruits)
fruits.pop()  # 마지막 값을 뗀다
print(fruits)
fruits.pop(4)  # 4번쨰 index값을 뺸다
print(fruits)

print('-'*40)
#list를 이용한 stack
#Stack메모리 컵구조 Last in First out ex)애플리케이션의 메인 메모리
#끝을 알때사용하면 더 빠름
stack= [3,4,5]
stack.append(6)
stack.append(7)
print(stack)
stack.pop()
print(stack)
stack.pop()
print(stack)
stack.pop()
print(stack)
stack.pop()
print(stack)
stack.pop()
print(stack)
#stack.pop() #데이터가 더이상 없음...
# #IndexError: pop from empty list

print('-'*40)
#Queue메모리 메모리에 데이터가 줄을 서서 입력됨
#output도 순서대로 나간다 !!
#First in First out 구조 ex) inputstream, outputstream
from collections import deque
queue=deque(['a','b','c'])
print(queue)
queue.append('d')
queue.append('e')
print(queue)
queue.popleft() #popleft()왼쪽부터 제거한다()안에 index인자 들어갈수 없음
print(queue)
queue.popleft()
print(queue)
queue.popleft()
print(queue)
queue.popleft()
print(queue)
queue.popleft()
print(queue)
#queue.popleft()

#comprehension   #iterable 을 iterator로 실체화
#1
suqares=[]
for x in range(10):
    suqares.append(x**2)
print(suqares)
#2 람다식을 포함
suqares2 = list(map(lambda x:x**2,range(10)))
print(suqares2)
#3
squares3 = [x**2 for x in range(10)]
print(squares3)
# 3가지 다 같은 값이 출력

a=[(x,y) for x in [1,2,3,] for y in [3,1,4] if x!=y]
#comprehension은 뒤에서부터 해석해보기
print(a)

b=[]
for x in [1,2,3]:
    for y in [3,1,4 ]:
        if x!=y:
            b.append((x,y))
print(b)

vec =[-4,-2,0,2,4]
[x*2 for x in vec]


c1= []
for x in (vec):
    x=x*2
    c1.append(x)
print(c1)

[x for x in vec if x>=0]
c2=[]
for x in vec:
    if x>=0:
        c2.append(x)
print(c2)

print([abs(x) for x in vec])
c3=[]
for x in vec:
    c3.append(abs(x))
print(c3)

freshfruit=[' banana ',' loganberry ',' passion fruit ']
print([weapon.strip() for weapon in freshfruit])
#strip() 는 문자열 앞뒤의 띄어쓰기를 제거해줌
freshfruit_1=[]
for weapon in freshfruit:
    a= weapon.strip()       #밑에랑 합쳐서
    freshfruit_1.append(a)  #freshfruit_1.append(weapon.strip()
print(freshfruit_1)

print([(x,x**2)for x in range(6)])
d=[]
for x in range(6):
    d.append((x,x**2))
print(d)

vec=[[1,2,3],[4,5,6],[7,8,9]]
print([num for elem in vec for num in elem])
h=[]
for elem in vec:
    for num in elem:
        h.append(num)
print(h)

# print([x,x**2 for x in range (6)])
# tuple로 x와 x**2를 묶기위해서는 (x, x**2)로 해야한다!!

#중첩 리스트 컴프리헨션

#list vs dictionary
list1 =[1,2,3,4,5]
dic = {'a':'1','b':'2','c':'3','d':'4',1:5}

# print(list1.__dir__, '\n', dir(list1))
# print(dic.__dir__, '\n', dir(dic))

from math import pi
print([str(round(pi,i))for i in range (1,6)])
pi_1=[]
for i in range (1,6):
    pi_1.append(str(round(pi,i)))
print(pi_1)

matrix =[[1,2,3,4],
         [5,6,7,8],
         [9,10,11,12]]

print([[row[i] for row in matrix]for i in range(4)])
matrix_1=[]
for i in range(4):
    for row in matrix:
        matrix_1.append(row[i])
print(matrix_1)

list2=list(zip(*matrix))
print(list2)
del list2[0]
print(list2)

del list2

tuple = 1,'2',3,4,5
print(tuple)
tuple2 = ('1',2,3,4,5)
tuple3=tuple,tuple2 # 중첩
print(tuple3)

list4 =["1",2,3,4,"5"]

#tuple의 크기
empty = ()
aTuple = 'hello',  #,로 되어있어 사이즈는 1이나 튜플이다
print(len(empty))
print(len(aTuple))

a,b,c,d,e = tuple # tuple = 위에 5개가 존재
#일대일 매칭이 가능해진다
print(a,b,c,d,e)
print(type(b)) # str 이지만 ''인식이 x
print(type(a))

#set: 순서, 중복 모두 불허
set1=[1,2,3,4,'5']
print(set1)
# set1[0]=7  불가능 마찬가지로 inex 교체 불가

a = set('abracadabra')
b= set('alacazam')
print('-'*40)
print(a)
print(b)

print(a-b) #말그대로 a의 원소에서 b의 원소제거
print(a|b) #a와 b의 합집합
print(a&b) #교집합
print(a^b) #차집합
print({x for x in 'abracadabra' if x not in 'abc'})

#####################################
#[]는 list    초기화 [],list()
#{}는 set     초기화 set()
#()는 tuple   초기화 (), tuple()
#{:} key:value  Dictionary다 초기화 {}, dict()

#dictionary
tel={'jack':4098,'sape':4139}
#dict-append()하는방법은 아래와 같음
tel['guide']=4127
print(tel)

#검색
print(tel['jack'])
#삭제
del tel['sape']
#정렬된 keyd의 set
print(sorted(tel))

print('guide'in tel)
print('jack'not in tel)

#dict()함수는 list[tuple()]을 이용하여 dictionary를 생성
b= dict([('sape',4139),('guide',4127),('jack',4098)])
print(b)

#키워드 인자를 통해 dictionary생성가능
c=dict(sape=4139,guide=4127,jack=4098)
print(c)

knights ={'gallahad':'the pure','robin':'the brave'}

for k,v in knights.items(): #items():원소의 목록
    print(k,v)

print(knights.keys()) #keys -> set구조, keys()->list
print(knights.values())

#enumerate() :인덱스와 값을 가지고 있다
list= ['tic','tac','toc']
for i,v in enumerate(list):
    print(i,v)
#0 tic
#1 tac
#2 toc