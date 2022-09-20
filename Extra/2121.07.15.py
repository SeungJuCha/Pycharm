# List 제일 중요한 부분
# list : 집합 데이터, 여러 데이터 값을 하나의 변수로 관리
# date type-> X 누군가 만들어야 하기에 datetime 라이브러리 존재
# 숫자로 저장을 하고 싶기에 사용한다 그리고 같은 변수로 지정시 변환되는 것을 관리
####관리 =CRUD-create,retrieve,update,delete
# year = 1999
# month = 2
# day = 1
# myBirth = [year, month, day ]
#
# year = 1999
# month = 2
# day = 22
# fdBirth = [year, month, day]
# print(myBirth[1],fdBirth[1])
# print(myBirth, fdBirth)

#for -in 반복문 사용방법  #while 반복의 차이점 이해하기!!!!
#for 단일값 변수 in 집합변수 :
#  띄어쓰기후 할일
#반복을 벗어나 할 일
# print("내생일은", end=":")
# for a in myBirth :
#     print( a, end=".") # 가로로 출력하면서 점을 찍고 싶을떄0
# print() #구분해서 아랫줄 윗줄 나누고 싶을 때
# for a in fdBirth :
#     print(a, end="/")
# print()

#교재 예제
# squares = [1,4,9,16,25]
# sq = squares[:] # 같은 리스트를 복사 (완벽한 복사 값 까지 )
# sq[0] = 100 #변환이 됨 복사를 해서 바꿀수 있다
# print(sq)
# sq1 = squares  # 주소를 복사 한다고 표현 리스트 이름이두개다
# print(sq1)
# sq1[0] =100
# print(sq1)
# squares[0]=100
# print("squares", squares)\
s=[1,2,3,4,5]
s_copy=s[:]
sq=s
s_copy[0]=100
print(s_copy)
print(s)
print(sq)
sq[0]=100
print(sq)
print(s)
s.append(12)
s[0:3]=[3,2,1]
print(s)

# a=1
# a라는 상자에 값이 들어가는게 아니라 값에 꼬리표가 붙는다고 생각
# # python 에서는
# b=a #a꼬리표가 붙어있는 값에 b꼬리표를 추가한다
# a=2 #2 에 a꼬리표 부착
# print(a,b)

# cubes=[1,8,27,65,125]
# cubes[3]= 4**3
# print(cubes)
#
# cubes.append(216) # append는 리스트 마지막에 이값을 추가
# cubes.append(7**3)
# print(cubes)

# letters=['abc','bcd','cde','def','efg','fgh','ghi']
# letters[2:5]= ['C','D',"E"]
# print(letters)
# letters[2:5]=[] #2이상 5미만의 값을 없앤다
# print(letters)
# letters[:]=[]   #전체  내용만 초기화(칸제외)
# # #letters = 0 도 가능하나 칸까지 없애버린다
# print(letters)
# print(len(letters))

# 리스트 중첩이 가능 2차 배열은 [[],[]] 이런식 index[][]사용가능
# a= [['a','b','c'],]
# b= [1,2,3]
# a.append(b)
# print(a[0][2]) # append를 사용한 2차 list 만들기
# a = ['a','b','c']
# x=[a,b]
# print(x)
# print(x[1][1])  # 새로운 변수를 사용해 2차 list 만들기
# a=[[3,4,5,6],[5,6,7,8],]
# a.append([7,8,9,0])
# for i in range(len(a)):
#     print (a[i],end='\n')

#제&분기 프로그래밍
# a_1=int(input('값을 입력하시오'))
# a_2=int(input('값을 입력하시오'))
# a_3=int(input('값을 입력하시오'))
# a_4=int(input('값을 입력하시오'))
# a_5=int(input('값을 입력하시오'))
# a_6=int(input('값을 입력하시오'))
# a_7=int(input('값을 입력하시오'))
# a =[a_1,a_2,a_3,a_4,a_5,a_6,a_7]
# if len(a)< 5:
#     print(a)
# elif len(a)==5:
#     a.append(7)
#     print(a,len(a))
# else:
#     a_copy=a[:]
#     a_copy.append(1)
#     print(a_copy, end='\n')

for i in range(1,10) :
    i+=1
    a_1= i+2
    print(a_1)

a=1
while a<10:
    a=a+1
    a_1 =a+2
    print(a_1)
# for i in range




# 조건문 & 반복문
# a = 0
# n = 12
# for a in range(n):  # range()는 ()안의 값 미만까지의 범위
#     b = 2
#     a = a**b
#     if a < 100:
#         print(a, end=',')
#     else:
#         print('100 보다 큼', end=',')
# print('\n#############################')
# a = 0
# n = 10
# while a < n:
#     print(a)
#     a = a++1  #++숫자는 그수에 계속해서 이걸 더한다

###수업 내용 # 조건문
# if 논리값 :
#     참일 경우 수행
# elif 논리값 2 :
#     거짓일 경우 또다른 조건을 판단하여 참일경우
# else :
#     거짓일 경우 수행할 일
# a = 0
# b = 1
# c = b > a  #True의 논리값
# if b > a:
#     print(b)
# else:
#     print(a)
# 반복문 while
# while 조건(논리값) :
#   참일 경우 수행 할 일
# 거짓인경우 while을 벗어나 일반 문장 수행
a = 0
b = 10

while a < b:
    a= a+1  #또는 a+=1
    print(a)
print('while 반복문 완료')

#피보나치 수열
a,b = 0,1 #다중 대입 파이썬만가능
while a<10:
    print("a=",a,",b=",b)
    a,b = b,a+b # 같은줄에서 동시의 일을 행하기 떄문
    # a=b   줄이 나눠진경우에는 값이 변해서 다음줄로 이행이 되기에
    # b=a+b

# x = int(input("please enter an integer:"))  # input 함수 사용자가 직접 지정할수 있음
#
# if x < 0:
#     x = 0
#     print('Negative changed to zero')
# elif x == 0:
#     print('Zero')
# elif x == 1:
#     print('Single')
# else:
#     print('More')

# 일반 언어 -조건문        성능      사용하는 곳
# if-else if -else      각 조건이 참인경우가 어느한 쪽으로 치우칠떄
#                    if가 대다수 확률 else if가 그다음 else가 라스트
# switch -case            각 조건이 참인경우가 비슷한 확률이거나
#                                알수 없을때
# 삼항연산자               2등
# 논리곱/논리합             1등

#반복문은 while밖에 없다
# for문은 한계를 안다(즉 몇번 일을 수행해야하는지 이미 안다)
# 그만큼 복제를 해서 한번에 수행해버림 다시 리턴해서 수행하는게 아님!!

#for(초기값; 조건값; 증감값) ->일반언어 (index로 반복)
#for-in                  -> 파이썬 (index를 모른다)

users = {'홍길동':'inactive', '임꺽정':'active', '장길산':'inactive'}
# dicionary
# for user, status in users.items():
#     if status =='inactive':
#         del users[user]
# print(users)

for user, status in users.copy().items(): #users.copy()는 user를 복사하고 user와 status로 일대일 매칭
                                        # 복사후 안에서 제외하는 법
    if status =='inactive':
        del users[user]
print(users)
#
#
#
active_users ={}                    # 새로운 dictionary를 생성해서 그안에 해당값을 대입
for user, status in users.items():  #user 와 status 에 이름과 in/active를 배정하는 과정
    if status == 'active':
        active_users[user] = status
print(active_users)

#range(초기값, 종료값, 증감값)
# a = range(-100,-10,30)
# for b in a:
#     print(b)
# print('range():',a)
# print('range()->list:', list(a))
#
# a=['Mary','had','a','little','lamb']
# for i in range(len(a)):
#     print(i,a[i])

# for n in range(2,10):   #2,3,4,5,6,7,8,9 까지다 미만이다 범위는 이상 미만이다
#     for x in range(2,n):
#         if n%x==0:
#             print(n,'equals',x,'*',n//x)
#             break
#     else:
#         print(n,'is a prime number')
# for b in range(2,10):
#     if b%2==0:
#         print(b,'equals',2,'*',n//x)
#     else:
#         print(b,'is a prime number')
#
# for n in range(2,10):   #2,3,4,5,6,7,8,9 까지다 미만이다 범위는 이상 미만이다
#     for x in range(2,n):
#         if n%x==0:
#             print(n,'equals',x,'*',n//x)
#     else:
#         print(n,'is a prime number')
for num in range(2,10):
     if num %2==0:
        print("Found an even number",num)
        continue
        # print("Found an even number",num)

# for num in range(2,10):
#     if num % 2 == 0:
#         print("Found an even number", num)
#
for num in range(2,10):
    for x in range(2,num):
        if num%x ==0:
            print("Found an even number",num)
        continue




# for num in range(2, 10):
#
#     if num % 2 == 0:
#         print("Found an even number", num)
#     else:
#         print("Found an odd number",num)