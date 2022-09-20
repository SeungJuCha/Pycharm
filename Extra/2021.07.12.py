tax = 12.5/100
price = 100.5
TaxValue = tax*price
total = price +TaxValue  #스크립트에서 _는 사용할수 없다! _는 이전값을 의미한다
                        #그래서 변수지정을 통해 이전값을 지정하고 변수로 표시를 해줘야 된다 (스크립트경우)

print(total)
print(round (total,2))  #round(값,숫자 ) 그 값을 소수점 숫자 자리에서 반올림이다

#쉬프트 연산 !!! 중요한거 한계점 계속 나누다 보면 0으로 표현이 되버리는데 이게 오류다!!! 정수부분까지만 나눠진다
#예를 들면 8을 8까지로는 나눠지지만 16이상으로 나눠버리면 0으로 표현되지만 이건 오류다! 실수연산이 불가능!
#

# a =1
# print(a<<1)  #<<는 곱하기 2이다 #2의 1승
# print((a<<1)+a)
# print(a<<2)  #2의 제곱
# print((a<<2)+a)
# print((a<<2)+a+a)
# print((a<<2)+a+a+a)
# # print((a<<3)-a)=7 랑 위의 값이 same
# # print((a<<2)+a+a+a+a)  랑 밑에값이 same
# print(a<<3)  #2의 세제곱

# a=8
# b=a>>3  #>>는 나누기 2의 세제곱!!
# print((a>>1)+b+b+b)
# print((a>>1)+b+b)
# print((a>>1)+b)
# print(a>>1)
# print((a>>2)+b)
# print(a>>2)
# print(a>>3)
# print(a>>0)

# a=1
# print(a<<31)
# print(a<<63)
# print(a:=a<<1024)

# print(a>>31)

a= -1
print(a<<1)
print(a<<2)
a=-2
print(a>>1)
print(a>>2)
# -2를 4로 나눌수가 없음(커지는건 한계가 없지만 나눠서 작아지는건 한계가 존재)
#파이썬 정수의 자릿수는 원칙적으로는 무제한이나 일반 프로그램에서 제한이 있기에 충돌이 일어난다

#bit 연산!!!
print("bit operator . . .")

a=1
print(~a)  # not
print(a^a) # xor

a=13 #00000000000
# #맨 처음 부분을 sign비트로 하는데 부호를 나타낸다 0은 + /1은 -
b=-15  #1111111111111
print(a|b) #or, 논리합 : 하나라도 1이 있으면 1
print(a&b) #and, 논리곱 : 둘다 1이여야 1
print(a^b) #xor, 배타적 논리합 : 두수가 같으면 0, 다르면 1
#계산을 할때 비트 자리 하나하나를 비교하면서 체크해간다
#0000000000
#1111111111


#문자열
# txt ='span eggs' #파이썬에선 '' "" 상관이 없음
# print(txt)
txt2 = 'doesn\'t'  #이런식으로 표시할떄 신경을 써야됨
# # #'를 \'로 표시함으로써 나는 문자로 인식해달라는 분리역할
# print(txt2)
txt3 = "doesn't" # 어퍼스트로피 인식을 위해 큰따옴표로 분릴를 한것
# print(txt3)
# txt4 = '"Yes," they said.' #""를 문자로 인식
# print(txt4)
# txt5 = "\"yes,\"they said."
# print(txt5)
#
# txt6 = 'First line.\n second line.'
# #\n 은 줄바꾸기
# print(txt6)

print('C:\some \name') #/n은 줄바꾸기  엔터키느낌
print(r'C:\some\name')  #특수기호를 안쓰기 위해 문자열 앞에 r을 대입
print('C:\some\\name') #아니면 \\로 통해 특수기호로 만드렁 버리기
print("""
Usage: thingy [OPTIONS]
-h Display this usage message
-H hostname Hostname to connect to""")  #문자열이 3줄이라서? "가 3개인건가??
print('''\
Usage: thingy [OPTIONS]
-h Display this usage message
-H hostname Hostname to connect to''')

# 연산자 오버로딩
# print('he'+'llo')
# print('he'+2*'l'+'o')
# # 문자열에서 *는 그만큼 반복이다
# print('he''llo')
# print('he', 'llo') # , 는 한칸 띄어씌기
#
# print(5*'hello')
#
# print("'he'+'llo'") # ""안 내용을 전부다 문자로 인식 그대로 프린팅
# print('he + llo') # 위와 마찬가지
# print('\'he\'+\'llo\'') #이것도 \를 그대로 두면 문자로 인식 --> 특수기호로 변환
# # 출력시 'he' + 'llo'로 나오는 이유?
# print(r'\'he\'+\'llo\'') #r로 시작을 했기에 특수기호 인식을 못한다!

#문자열 = 문자들을 열순으로 표기 = 문자 + 배열
s= "hello"
#일반언어 [] : 배열
#python [] : List, 함수로는 list()
# print(s[0]) #s의 첫번쨰 문자를 print해라
# # #[0]=[-5] 와 동일 앞이냐 뒤기준이냐의 차이 INDEX라 칭하고  0시작
print(s[-4])
# print(s[:-2]) # [시작 : 끝] 시작 부터 끝까지 를 인식 대신에 0은 표기를 안해도 됨
# print(s[-2:])
# print(s[0:-2])
print(s[:4])
print(s[:-1])
print(s[-5:-2])
print(s[4:2])
print(s[4:4])
#문자열이 저장이된후 문자 변환은 불가하다
# s[0]='J'
# print(s)
# TypeError: 'str' object does not support item assignment

# s='asdfasdfasdfasdfasdf'
# print(len(s))  #len는 길이 수

# s= 'python1'    #*5
# print(s[:3]+ s[3:])  #0부터 시작이라는 점
# print(s[:-3]+s[-3:])
# print(s[:3], s[3:]) # 3인경우 0부터 3미만 (이상 미만)이라는것
# print(s[:-3], s[-3:])
#범위 박의 인덱스인 경우
# print(s[33])   # IndexError: string index out of range
#앞에다가 범위를 잡아줘야 된다!!! 오류는 안나되 어찌 잡냐에 따라 출력이 없을수 있음

# print(s[-33:-3]) #-3>-33 이라는거 항상범위를 지정할떄 작은거에서 큰거로 지정



a=1
b = bytearray(bytes('aaaa', 'UTF-8'))
#b[1] = 'b'

#풀어서 설명하면 밑에 부분이다

# s = '한글'
# print(s)
# # s--> 출력 -->한글 깨짐
# b = bytes('aaaa', 'UTF-8')
# #b를 byte로 변환, 제대로된 문자셋을 사용하여 변환 (UTF-8)
#
# print(b)
# ba = bytearray(b) #문자에 삽입 또는 변환을 할때
# print(ba[1])
# print(ba)
# s = str(ba)
# print(s)
#
#
# #숫자의 특징 문자와의 다른점
# c = 123
# d = '123'
#
# #print(c[0])  #c는 수이기 때문에 쪼개는게 불가능
# #c[0] = 5    #[0]에 해당하는 즉 1을 5로 바꾸는거 말도 안됨
# print(d[0]) #1이 출력됨


# s=[1,2,3,4,5]
# s_1=bytes(s)
# s_2=bytearray(s)
# # s_1[1]=3 #TypeError: 'bytes' object does not support item assignment
# print(s_2)
# s_2[1]=3
# print(s_2)
# print(s_1)

a='hell0'
aa=a.encode()
a_1=bytes(aa)
a_2=bytearray(aa)
print(a_2)
# a_1[1]='k' #TypeError: 'bytes' object does not support item assignment
H='k'
aaa= H.encode()

a_2[1]=(aaa)
print(a_1)
print(a_2)