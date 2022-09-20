b1 = True #  boolean True or False
print(b1)
print(type(b1))
print(id(b1))

b2 = 100<10
print(b2)

print(not b1)
print(b1 and b2)
print(b1 or b2)
x1 = 1
x2 =2 ; x3=x2 ; x4 = x1+x3

print(id(x2))
print(id(x3))
print('x2의 id :{}, x3의 id {}'.format(id(x2),id(x3)))
print(f'x2의 id :{id(x2)}, x3의 id {id(x3)}')

y1 = 1.145161
print('y1 = {0:.4f}'.format(y1)) #소수점 반올림 round

y2 = int(y1) # y1의 소수점 버림 후 프린팅

print(100 // 9) # 몫
print(100 % 9) # 나머지
print( 100 **3) # n 승(
z1 = complex(1,-2) # 복소수 표현 (x + yj) 형태 x,y 표현
print(z1)
print(z1.__class__) #클래스 표현법
print(type(z1))

#isinstance 클래스로 만들어낸 다른 객체
#calss = 설계도

print('-' * 30)
a = input('학교 :') ; b = input('학부 :')
c = input('학번 :') ; d = input('이름 :')

Student_info = (a,b,c,d)

def Information (*Student_info):
    for item in Student_info :
         print(item, end = '\n')
    print('-'*30)

Information(*Student_info)



