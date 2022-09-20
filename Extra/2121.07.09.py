#literal: 고유의 값(의미)
#값 - 데이터 - 형태
# 1, 1.0, '1', "1", "하나", "한 개"
# 대표= 1 이다. but 형태가 다양
# a = 1; a는 변수
# const a = 1; a는 상수 (변하지 못함. 즉 다른값을 대입시 오류)

# := 대입 표현식
a = 1
b = 2
c = a+b
d = c-b+a
d = (c:=a+b) - b+a

# #  논리연산
a =False
c = a or b # 둘중 더 먼저 참인 값
d = a and b# 둘중 하나라도 false 면 그냥 false
e = not a # not false= True
print(d)
#
# f= a or (b:=not a) #b값이  not a 라고 저장데이터를 바꿔버림
# print(f,b)
#
a= True
f= a or (b:=not a)
print(f,b) # a값이 참이므로 뒷값을 볼필요가 없다 즉 계산이 그대로 끝나버림
# # 결론 b:= not a는 필요가 없어버려 생각을 안함 그래서 변수 변환 X
# # or의 특징: true가 나올때까지 계산-----> not-if 조건문 실행 가능
# a= True
# f= a and (b:=not a)
# print(f,b) # and는 true true 여야 true 하나라도 false 면 false



# import keyword
# print(keyword.kwlist)

# import numpy
#
# print(numpy.__version__)

