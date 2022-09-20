import numpy as np
import matplotlib.pyplot as plt

#히스토그램
# mu, sigma = 100,15 #mu 평균 sigma 표준편차
# rn=np.random.standard_normal(1000)
# print('standard_normal :')
# print(rn)
# print('np,random.randn(10000) : ',np.random.randn(1000))
# # 정규분포도 난수 발생 : randn
# data = mu + sigma * np.random.randn(10000)
# data = mu + sigma * np.random.standard_normal(10000)
# plt.hist(data)

#그래프 옵션 설정
# x1 = np.arange(20)
# x2 = np.power(x1,2)
# x3 = np.power(x1,3)


# plt.plot(x1, color = 'red', marker = 'o', linestyle = '-')
# plt.plot(x2, color = 'blue', marker = 's', linestyle = '--')
# plt.plot(x3, color = 'green', marker = '^', linestyle = '-.')

# plt.title('Title')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()  # x y 축 밑 제목 집어넣는 법
#
# plt.xticks(np.arange(0,20, step =4))   #x y 축범위 조정
# plt.yticks(np.arange(10,10000, step = 1500))
#
# plt.legend(['x1','x2','x3'])
# for f in [x1,x2,x3] :
#     for i , data in enumerate(f) :
#         plt.text(i,data +1.5, data)
# plt.style.use('ggplot')  #눈금 추가 느낌
# # plt.plot(x1)
# # plt.plot(x2)
# # plt.plot(x3)
# print(plt.style.available)

# x= np.arange(40)
# plt.subplot(2,2,1)
# plt.plot(x,x)
# plt.subplot(2,2,2)
# plt.plot(x,x**2, color= 'red')
# plt.subplot(2,2,3)
# plt.plot(x,x**3,color = 'blue')
# plt.subplot(2,2,4)
# plt.plot(x,x**4,color = 'green')

#이미지 출력
# img = plt.imread('C:\\Users\\chasj\\Desktop\\olympic.jpg')
# print(img)
# plt.imshow(img)
# plt.show()

#########pandas#############
#시간 계열 데이터셋
import pandas as pd
# data = [1,2,3,4]
data = [(1,2,3,4),(2,3,4,5),(6,7,8,9),(11,12,13,14)]
# 집합데이터는 차원계산시 유의(스칼라 데이터)
#튜플도 가능함 대신 datatype 이 object
#index 는 row의 이름 column &label은 세로의 이름
index = ['A','B','C','D']
#인덱스에는 문자와 시간을 사용가능(date)
#Series & DataFrame만 존재 각 1차 2차원이다 in pandas
k = pd.Series(data = data, index = index) # 키워드 인자
# l = pd.Series(data,index) # 위치 인자
print(k)

print(k.get['f']) #오류 안남
print(k['f'])  #오류
#Data Frame : Series(colum 명 : 데이터셋의 튜플)와 인덱스로 만든다
data = {'colum1':[1,2,3,4],'colum2':['a','b','c','d']}
c= pd.DataFrame(data,index)
print(c) #만들때는 가론데 세로로 입력 끼워 맞춘느낌

#Data Frame : row(colum값들의 집합)와 colum명으로 만든다
data = [[1,'a',False],
        [2,'b',True],
        [3,'c',False],
        [4,'d',True]]
columns = ['column1','column2','column3']

#rows = ['a','b','c','d']
d = pd.DataFrame(data =data,index = index,columns =columns)
#row = rows 데이터 프레임에 row라는 인자는 없다 그래서오류
#or (data,index,columns) 키워드인자
print(d) #가로 따로 세로 따로 합치기! 좀더 이해가 편하다 매칭

#numpy의 ndarray사용
data = np.zeros((4,2))
g = pd.DataFrame(data)
print(g)

data = [
        ['Sun', 10 , None],
        ['Mon',0, None],
        ['Tue',15,None],
        ['Wed',3,'Children\'s Day'],
        ['Thur',100, 'Birth Day'],
        ['Fri',200, None],
        ['Sat',7, 'Parent\'s Day']
]
index = pd.date_range('20210726',periods=7) #date time index
column = ['Date','Budget','Anniversary']
H = pd.DataFrame(data,index,column)
print(H)

file = 'C:\\Users\\210821A\\Downloads\\titanic_train.csv'
df = pd.read_csv(file)
# print(df)
# print('-'*40)
# print(df.head())  #처음 데이터
# print(df.tail()) # 끝 데이터
# print(df.info())
# print('-'*40)
# print(df.isna())  #missing값 표시
# #갑이 없으면true 있으면 False로 나온다
# print('-'*40)
# print(df.describe()) #통계
# print(df.describe(include = 'all'))
# print(df.corr()) #열간의 상관관계
# print(df.mean)
#
# print('*'*40)
# print('\n')
#
# print(df.groupby('Survived').mean())
# print(df.groupby(['Sex','Survived','Embarked']).mean()) #대분류 ->소분류
#
# print()
# print(df['Survived']) #원하는 부분 도출
# print(df.Survived)
#
# print()
# print(df.loc[0]) #특정 위치 데이터 indexing
# print(df.loc[0,'Name'])#특정 위치의 특정 데이터
#
# print()
# print(df.iloc[0]) #순서로위치를 지정
# print(df.iloc[0:5,1:4])  #0부터 4까지의 데이터에서 1부터 3까지의 세로줄만 도출

print('\\'*80)
print('\\'*80)

print(df[['Name','Age','Sex','Survived']])

mask = (df.Sex == 'female')
print('mask :\n' ,mask)
print('df[mask]\n :',df[mask])#여성인 경우만 가져오기

df['new']= 0  #새로운 'new'라는series 추가 값은 0
print(df.head())

df['family']= df['SibSp'] +df['Parch']
#데이터의 합을 새로운 이름으로 지정
print(df.head(10))

df1 = df.drop(labels ='new',axis =1)  #컬럼 삭제
df.drop(columns = ['new','family'],inplace = True)
#inplace사용시 반환값이없는대신 원본 수정이 가능 return 값은 없음
print(df.head())

print('new 컬럼삭제:',df1.head(2))
print('new,family(inplace) 컬럼 삭제:',df.head(2))

print('-'*100)
#z컴럼 세로로 붙이기   axis =1
np.random.seed(0)
data = np.random.randn(len(df))
standard = pd.Series(data,name = 'standard')
print(standard)
total = pd.concat(objs=[df,standard],axis=1)
print(total)

print()
#z컴럼 가로로 붙이기   axis =0 방향
data = np.arange(len(df.columns)).reshape(1,-1)
number = pd.DataFrame(data,columns=df.columns)
print(number)
df2 = pd.concat([df,number],axis=0)
print(df2.tail(3)) # 제일 먼저와야되는것이 맨뒤에 붙기에 꼬인다
#     PassengerId  Survived  Pclass  ...   Fare Cabin  Embarked
#889          890         1       1  ...  30.00  C148         C
#890          891         0       3  ...   7.75   NaN         Q
#0              0         1       2  ...   9.00    10        11


print(df2.reset_index(drop = True)) #index 라벨 재설정
#drop = True 기존걸 날려버린다
print()
df = df.rename({'Survived':'생존'},axis = 1) #라벨 이름 바꿈
print(df)
print(df.info())

print('#'*40+'\n'+'#'*40)

#fillna 측정이 안된 값을 채워넣는거 대신 그런 결측값이 많을 경우 삭제
print(df.Age) #888     NaN
age_mean = df.Age.mean()
print(age_mean)#29.69911764705882
age_fill =df['Age'].fillna(value = age_mean)
#= df.Age.fillna(value = age_mean)
print(age_fill) #888    29.699118

#element wise함수
#applymap -dataframe 전체
def f(x):
        return len(str(x))
print(df.applymap(f))
#applymap은 함수를 인자로 받아서 DataFrame의 !개별! 원소에 대하여 함수를 실행한다

#map seires의 개별값에 접근해 변경하는 방법
print(df.Sex.map(lambda x:0 if x=='female'else 1))
# print(df.Sex.map({'female':0,'male':1}))

#apply 함수가 1개이상 인자가 필요할때
#처음 인자는 dataframe의 값에 고정
# #나머지 인자는 위치나 keyword로 전달
def n_square(x,n):
        return x**n

print(df[['Age','SibSp','Fare']].apply(n_square,args = [2]))
# 위 3개가 항상 효율적이지는 않다
#함수가 벡터화 연산을 지원할경우 함수를 그래도 쓰는게 더 빠름

#####DataFrame저장 #######
df.to_csv('titanic.csv')
