#class --user define type
# 연산자 오버로딩--속도 저하 -->함수를 만들어서 대체
#그래서 남이 만든 라이브러리를 쓰자
################################
#collection (콜렉션)  파이썬 용어(compound) :관리와 사용
#주머니라 생각하면 단순하다(원하는것을 꺼내기가 힘듦)
#그래서 순서를 제공함으로써 원하는것을 얻기 쉽게함
#List 는 순서(index)가 존재,중복 가능
# Set 는 중복 불허, 순서도 없음
# 사용 : List -> 중복 데이터가증가 -> Set를 사용
#사용이 쉽다= 찾기 편하다 숫자보다 이름(문자열)이 편함
# Map(Dictionary)
# 사전 - 뜻(content,value) <- 단어(keyword)
# key-value의 쌍(튜플:tuple) 으로 존재한다!!!


#함수 : 작은 프로그램
#def 함수이름 (인자들) :
# 코드
a1 = 1

def a () : #인자b = a+1를 쓰지 않아도 ()를 필요
    return a1+1  #에러난 이유 a가 함수를 표시해버려서
#그래서 이왕이면 함수명을 변수와 다르게 써라!!

print(a1)
print(a())

def b(a) : #인자를 사용했기에 프린트 시에도 인자값에 대입필요
    return a**2
print(b(a1))
#print(b())  TypeError: b() missing 1 required positional argument: 'a'
# positional argument <-> keyword argument
print(b(a=5))

def c(a, input) :
    return a** input
print(c(3, input =2))
#Positional argument after keyword argument 순서가 잘못됨

def d(a, input=2) :
    return a % input
print(d(5))
print(d(5,3))
print(d(5, input=4))

#user = input( "입력하세요 :") #키워드 인자 사용 불가능

# def ask_ok(prompt,retries=4, reminder='please try again') :
#     while True:
#         ok= input(prompt)
#         if ok in ('y','ye','yes'):
#             return True
#         if ok in ('n','no','nop','nope'):
#             return False
#         retries = retries-1
#         if retries <0:
#             raise ValueError('invalid user response')
#         print(reminder)
#
#
# print(ask_ok('정말 끝내길 원하세요?'))

i =5
def f(arg = i):
    print(arg)

i=7
f()  # 7이 출력이 아니라 5가 출력이 되는데 이유는?
#def하는순간 arg=i=5로 값이 fix가 되버리기 때문에
#밑에서 바꾼다한들 의미가 없음

def f(a,L=[]):
    L.append(a)
    return L
print(f(1))  # L=[1]로 되버린것
print(f(2))
print(f(3))

def f(a,L=None):  #재정의: 기존함수를바꿔버림
                  # L값에는 어떠한것도 저장하지 않겠단 마인드
    if L is None: #재정의는 인자가 첨삭시 over-loading
                  # 상속의 경우 인자가 동일하면 over-riding# over-loading& over-riding 는 기존 함수가 살아있음
        L=[]
        L.append(a)
        return L
print(f(1)) #만약 여기서 L값을 설정하면 L이 change
print(f(2))
print(f(3))

def parrot(voltage,state='a stiff',action='voom',type ='Norwegian Blue'): #4가지 인자중 3개를 정의가 됨 voltage만 미지수
    print("--This parrot wouldn't",action,end =',')
    print(" if you put",voltage,"volts through it")
    print("--Lovely plumage, the ",type)
    print("--It's",state,"!")


#positional 인자는 말그대로 함수에 있는 인자순서대로 매칭
#keyword인자는 그 함수 인자중 내가 원하는 것을 픽해서 바꿀수 있음 순서바꿔서 지정해도 오류는 없다

# parrot(1000) # 1 positional argument
# parrot(voltage=1000) # 1 keyword argument
# parrot(voltage=1000000,action='VOOOOOM') # 2 keyword arguments
# parrot(action='VOOOOOM',voltage=1000000) # 2 keyword arguments
# parrot('a million','bereft of life','jump','Norwegian Red') # 3 positional arguments 인자 순서대로 3개를 지정
# parrot('a thousand',state='pushing up the daisies') # 1 positional, 1 keyword

#올바르지 않은 호출
# parrot() # required argument missing
# parrot(voltage=5.0,'dead') # 키워드 인자를 한번쓰면 그뒤에도 키워드인자로 대응시켜야된다 이미 순서가 꺠진거나 마찬가지이기에
# parrot(110,voltage=220) # duplicate value for the same argument 동일 인자 중복 포지션과 키워드인자를통해
# parrot(actor='John Cleese') # unknown keyword argument 없던 인자 생성

print("-"*40)
def cheeseshop(kind,*arguments,**keywords): #단일 인자, 리스트인자, dic인자
    print("--Do you have any",kind,"?")
    print("--I'm sorry, we're all out of",kind)
    for arg in arguments: #리스트 인자라서 여러개가 가능하다 집합데이터:collection
        print(arg)
    print("-"*40)
    for kw in keywords: #dic 인자는 여러개가 여러개 keyword = value
        print(kw,":",keywords[kw]) #keyword[kw]= kw변수의 value값
cheeseshop("Limburger","It's very runny, sir.",
           "It's really very, Very runny,sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")

#def f(pos1,pos2,/,pos_or_kwd,*,kwd1,kwd2) :
def standard_arg(arg): #키워드든 포지션이든 상관이 없음
    print(arg)
def pos_only_arg(arg,/): #position인자만 사용하는 함수
    #/앞에있는 인자는 모두다 위치인자만 사용해야한다
    print(arg)
def kwd_only_arg(*,arg): #keyword만 사용하는 함수
    #*뒤에있는 인자는 모두다 키워드인자만 사용해야한다
    print(arg)

standard_arg(1)
standard_arg(arg=2)
pos_only_arg(3)
#pos_only_arg(arg=3) 에러 발생(위치인자만 사용해야하는데 키워드인자를 써버림)
kwd_only_arg(arg=6)
#kwd_only_arg(6)   에러 발생(키워드로만 써야하는데 위치인자를 써버림)

#종합예제
def combined_example(pos_only,/,standard,*,kwd_only):
    print(pos_only,standard,kwd_only)

#combined_example(1,2,3) #3이 원인으로 에러발생
combined_example(1,2,kwd_only=3)
combined_example(1,standard=2,kwd_only=3)

#unpacking방법 묶음을 푼다 이런느낌
#dic()의 인자와 앞의 인자를 혼동
def foo(name,**kwds):
    return name in kwds
 #k={'name':'2'} 하는것과같음
 #foo(name=1,**{'name':2}) #**name: unpacking name=2라칭함 인자 충돌
def concat(*args,sep="/"):
    return sep.join(args) #args 리스트 인자를 출력하는데 sep를 추가시킨다
print(concat("지구","화성","수성"))
print(concat("지구","화성","수성","/"))
print(concat("지구","화성","수성",sep='-'))

year = str(1999)# 문자로 받아야 한다는점!!TypeError: sequence item 0: expected str instance, int found
month = "2"
day = "1"
myBirth = [year, month, day]
print(concat(year,month,day))
print(type(myBirth[0]))
print(concat(*myBirth,sep ='.'))  #리스트인자 unpacking 방법 *리스트변수
# print(concat(myBirth,sep="."))  TypeError: sequence item 0: expected str instance, list found

for i in range (1,5,+1):

    print(range(i,6))
    print(list(range(i,6))) #list의 범위를 3이상 6미만까지 출력 using리스트 함수
temp = [3,6]
#print(range(temp)) #오류 왜냐 리스트는 정수형태가 아니기에
print(range(*temp)) #temp 리스트를 unpacking한것
print(list(range(*temp)))

d={"voltage":"four million","state":"bleedin' demised","action":"Voom"}
#튜플로 dictionary생성 전달법 변수 ={"이름":"값"}
#유의점: 반드시 str 문자열로 집어넣어라 숫자여도
parrot(**d) #d를 unpacking을 통해 일대일 매칭

#lambda : 인라인 함수
def make_incrementor(n):
    return lambda x:x+n #x라는 인자에 n값을 더해서 리턴해라

f = make_incrementor(42) #n=42
print(f(x=0))  #x=0
print(f(1))  #x=1

pairs =[
    (1,'one'), #tuple
    (2,'two'), #pair[0]=2 pair[1]='two'
    (3, 'three'),
    (4, 'four')] #list
pairs.sort(key=lambda pair:pair[1]) #sort는 순서분류 오름차순 pair[1]은 (1,'one')의 뒤에것
# pair의 이름은 뭐든지 상관이 없음 단지 무엇을 기준으로 잡을지 숫자가 중요
print(pairs)

#1
def f(ham,eggs):
    print("Annotations:",f.__annotations__)  #__annotations__는 Class판별 :str, int ....
    print("Arguments:",ham,eggs)
    return ham+eggs
#2
def f(ham:str,eggs:str ='eggs')->str:
    print("Annotations:",f.__annotations__)
    print("Arguments:",ham.__class__,eggs.__class__)
    return str(ham) +'and'+str(eggs)

print(f('spam','eggs'))
print(f(1,2))
print(f(b'Hello,world!')) #b 문자열은 이 문자열을 byte로 고쳐라 이말
