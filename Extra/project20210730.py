import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from dateutil import parser
import pandas_datareader.data as web
from datetime import datetime


# 데이터를 적재하고 날짜를 인덱스로 지정
file = "D:\\pycharm\\IT innovation\\venv\\FremontBridge.csv"
file2 = "D:\\pycharm\\IT innovation\\venv\\BicycleWeather.csv"
counts =pd.read_csv(file, index_col ='Date', parse_dates = True)

weather = pd.read_csv(file2,index_col = 'DATE',parse_dates=True )
# weather2 = pd.read_csv('BicycleWeather.csv',index_col = 'DATE',parse_dates=True)
# print(counts.head(3))
# print(weather.head(3))

#일별 총 자전거 통행량을 께산해서 별도의 DataFrame에 넣음
daily = counts.resample('d').sum() #day기반(날짜) 리샘플링
daily['Total'] = daily.sum(axis =1) #새로운 열 추가
daily = daily[['Total']] #다른열 삭제 #2차배열이기에
# print(daily.head(3))

#요일 열 추가
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
for i in range(len(days)):
    daily[days[i]]=(daily.index.dayofweek == i).astype(float)  #날짜와 요일 맞추기 맞으면 1로 출력
# print(daily.head(7))

#휴일에 자전거 타는 사람
from pandas.tseries.holiday import USFederalHolidayCalendar #미국 공휴일 달력 사용
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012','2016') #2012에서 2016까지의 휴일
daily =daily.join(pd.Series(1,index = holidays, name ='holiday')) #holiday라는 열을 추가 내용은 holidays
#내용은 holidays의 내용 이름은 holiday로 axis=1에 추가
daily['holiday'].fillna(0,inplace =True)
#holiday라벨의 내용을 0으로 채우되 만약 실제로 휴일이라면 1로표현
# print(daily.head())
# print(daily['2013-7-1':'2013-8-1'])

#일조시간에 자전거를 타는 사람

def hours_of_daylight(date,axis = 23.44,latitude= 47.61) :
    #해당 날짜의 일조시간 계산
    days = (date - pd.datetime(2000,12,21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return  24 * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180

daily[['daylight_hrs']] = list(map(hours_of_daylight, daily.index))  #map(함수, 인자) 함수에 인자를 전부 대입후 리스트화
print(daily.head(3))
# daily[['daylight_hrs']].plot(color='Blue',linestyle ='--')
# plt.ylim(8, 17) # y축범위
# plt.grid()
# plt.show()

# 데이터에 평균 기온과 전체 강수량 추가
# 인치 단위의 강수량과 더불어 날이 건조했는지(강수량이 0) 알려주는 플래그 추가
# 기온은 섭씨 1/10도 단위, 섭씨 1도 단위로 변환
weather['TMIN'] = (weather['TMIN']/10)  #값들은 나누기 10과 동일
weather['TMAX'] /= 10
weather['Temp (c)'] =(weather['TMIN'] + weather['TMAX'])/2.0
# 강수량은 1/10mm 단위; 인치 단위로 변환
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)  #0일경우 1출력 아니면 0
daily = daily.join(weather[['PRCP', 'Temp (c)', 'dry day']])

# print(daily.head(15))

print('*'*50)
#첫날부터 증가하는 계수기를 추가해 몇해가 지났는지를 측정
daily['annual']= (daily.index - daily.index[0]).days /365.
# print(daily.index[0]) #2012.10.3
#index는 가로
# print(np.sum(daily['2013-8-1':'2013-8-5']))
# print(np.sum(daily['2014-8-1':'2014-8-5']))
# print(np.sum(daily['2015-8-1':'2015-8-5']))
# print(np.sum(daily['2016-8-1':'2016-8-5']))
# print(np.sum(daily['2017-8-1':'2017-8-5']))
# print(np.sum(daily['2018-8-1':'2018-8-5']))
#


# # 널값 행은 제거
# print(daily.tail(10)) #값이 없는 부분이 존재
daily.dropna(axis=0, how ='any', inplace =True) #그 NaN값을 채워주는 역할
column_names =['Mon','Tue','Wed','Thu','Fri','Sat','Sun'
               ,'holiday','daylight_hrs','PRCP','dry day','Temp (c)','annual']
print('NaN record removed\n',daily.tail(10))
temp_daily = daily.copy()
daily = daily['2013-1-1':'2014-12-31']
val_daily = temp_daily['2015-1-1':'2015-8-31']
x = daily[column_names] #label data
y = daily['Total']      # soulution
valx = val_daily[column_names]
valy = val_daily['Total']

from sklearn.linear_model import LinearRegression #최소자승법을 이용한 그래프
model = LinearRegression(fit_intercept =False)
model.fit(x,y)#기존 데이터학습
model.fit(valx,valy)
daily['predicted']= model.predict(x)
val_daily['predicted']= model.predict(valx)#결과에측
daily= pd.concat([daily,val_daily],axis = 0)
#총 자전거 통행량과 예상 자전거 통행량 비교
daily[['Total','predicted']].plot(alpha =0.5) #시각화 alpha는 투명도

plt.show()

# ___________________________________
# 각 특징이 요일별 자전거 통행량에 얼마나 기여하는지 추정하는 선형 모델 계수
params = pd.Series(model.coef_,index = x.columns)
# print(params)

# 불확실성에 대한 척도 없이는 해석이 어려움
# 데이터의 부트스트랩 표본 재추출(bootstrap resampling)을 사용하여 불확실성을 계산
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(x,y)).coef_
              for i in range (1000)],0)
# print(err[0:4])
print(pd.DataFrame({'effect':params.round(0), 'error':err.round(0)}))

