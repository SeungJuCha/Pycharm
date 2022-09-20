import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from dateutil import parser
import pandas_datareader.data as web
#파이썬 날짜와 시간
from datetime import datetime
# today =datetime(year =2021,month = 7, day =28)
# #today = datetime.today()
# print(today)
# lastday = datetime(year =2021, month =12, day =31)
# print('{}일'.format(today - lastday))
# #세계 표준 양식을 맞춰야 에러가 안난다
# date1 = parser.parse('15th of Nov,2021')
# date2 = parser.parse('2013-2-8')
# print(date1, '\t',date2)
# print(date1.strftime('%A')) #요일
#
# date = np.array('2021-07-28',dtype= np.datetime64)
# # date = np.datetime64('2021-11-15 12:00:30')
# print(date)
# dateArray = date +np.arange(10)#가장 작은 단위기준 리스트 생성
# print(dateArray)
# #######
# index = pd.DatetimeIndex(['2017-10-12','2017-11-12'
#                           ,'2018-10-12','2018-11-12'])
# data = pd.Series([0,1,2,3],index = index)
# print(data)
# print(data['2017-11-1':'2018-10-31']) #데이터 범위로 뽑아내기
#
# index = pd.DatetimeIndex(dateArray)
# data = pd.Series(np.arange(10),index = index)
# print(data)

# print('2021년 8월까지남은 날짜는 {}일'.format(len(data[:'2021-7'])))
# for key, value in data.items():
#     if (value == 5):
#         print('{}는{}일데이터'.format(value,key))
#         break


# start = datetime(2016,2,19)
# end = datetime(2016,3,4)
gs = web.DataReader('005930','naver')

print(gs.head())
# gs.info
# print(gs.index)
#
# plt.plot(gs['Close'],gs.index)
# plt.show()
###########################################
file = "D:\\pycharm\\IT innovation\\venv\\FremontBridge.csv"
data =pd.read_csv(file, index_col ='Date', parse_dates = True)

# print(data.head())
data.columns = ['East','West']
data['Total'] = data.eval('West+East')
# print(data.head(15))
# print(data.describe())
#
#전체 통행량만 비교
# seaborn.set()
# # data.plot()
# # plt.ylabel('count of bicycle/hour')
# # plt.show()
#
#3개를 한번에 비교하기 위함 w,e,t
# weekly = data.resample('W').sum() #주단위로 resampling
# print(weekly)
# weekly.plot(style = [':','--','-'])
# plt.ylabel('count of bicycle/week')
# plt.show()

#30일 이동 평균 rolling)
# daily = data.resample('D').sum()
# daily.rolling(30, center =True).sum().plot(style = [':','--','-'])
# plt.ylabel('mean hourly count')
# plt.show()

#Gaussian smoothing 적용
#롤링 평균을 부드럽게 표현 계산이 포함되면 데이터 왜곡 가능성이 높음!
# daily.rolling(50, center = True,win_type = 'gaussian')\
#     .sum(std =10).plot(style =[':','--','-'] )
# #윈도우 폭 50과 윈도우 내 가우스 폭 10 지정
# plt.show()

#하루 시간대를 기준으로한 함수로 평균 통행량 보기
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4*60*60*np.arange(6) #4시간 60분 60초 * 6개
# by_time.plot(xticks = hourly_ticks,
#              style = [':','--','-'])
# plt.show()

#요일별
# by_weekday = data.groupby(data.index.dayofweek).mean()
# by_weekday.index = ['Mon','Tue','Wed','Thurs','Fri','Sat','Sun']
# by_weekday.plot(style =[':','--','-'])
# plt.show()

#다중 서브 플롯
#주말과 주중의 시간대별 추이 비교시 좋음
weekend = np.where(data.index.weekday <5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()

# print(by_time)
fgs, ax = plt.subplots(1, 2, figsize=(18,5))

by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays',
                           xticks=hourly_ticks, style=[':','--','-'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',
                           xticks=hourly_ticks, style=[':','--','-'])
plt.show()

