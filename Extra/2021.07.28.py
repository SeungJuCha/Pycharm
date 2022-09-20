import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
####Seaborn########
import seaborn as sns

# print(sns.get_dataset_names())
iris =sns.load_dataset('iris')
titanic =sns.load_dataset('titanic')
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
#sns.rugplot
x =iris.petal_length.values
sns.rugplot(x)
plt.title('iris 데이터중 꽃잎의 길이에 대한 rugplot')
plt.show()

sns.kdeplot(x)
plt.show()

sns.distplot(x,kde =True,rug=True)


sns.countplot(x='class',data = titanic)
plt.title("number of passenger' no./class plot")


sns.countplot(x='day',data =tips)
plt.title('tip\'s count')

###다차원 데이터 플롯######

sns.jointplot(x = 'sepal_length',y='sepal_width',
              data = iris)#,kind ='kde')
plt.suptitle('length and width of sepal',y =1.02)

sns.pairplot(iris,hue ='species',markers =['o','s','D'])
plt.title('Iris Pair Plot')
plt.show()

# heat map
titanic_size = titanic.pivot_table(index = 'class',columns='sex',aggfunc='size')
print(titanic_size)
sns.heatmap(titanic_size,cmap =sns.light_palette('blue',as_cmap=True),
            annot = True,fmt = 'd')
# cmap은 heatmap종류
# annot는 annotation값을 써놓는 것 fmt = formatting
plt.title('Heatmap')
plt.show()

# 바이올린 형태의 그림
sns.violinplot(x='day',y='total_bill',data =tips)
plt.title('tip\'s count/day')
plt.show()

# stripplot 은 바이올린 형태를 점으로 찍어서 보여주는것
sns.stripplot(x='day',y='total_bill',data = tips,
              jitter = True)# 겹치는것을 없앤다
plt.title('tip\'s count/day')
plt.show()

# swarmplot
sns.swarmplot(x='day',y='total_bill',data = tips)
plt.title('swarmplot')
plt.show()

#boxplot
sns.boxplot(x='day',y='total_bill',data = tips)
plt.title('boxplot')
plt.show()

flights_passengers = flights.pivot('month','year','passengers')
plt.title('Heat map')
sns.heatmap(flights_passengers,annot = True,fmt = 'd',linewidths =1)
plt.show()

def sinplot(flip =1):
    x =np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
sns.set_style('darkgrid')
sinplot()
plt.show()

