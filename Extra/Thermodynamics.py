import numpy as np
from pyXSteam.XSteam import XSteam
s_t = XSteam(XSteam.UNIT_SYSTEM_MKS)
''' m/kg/sec/°C/bar/W'''

"""기존 문제, 573K, 6MPa, 1.4Mpa, 75kPa
    Regenerative Rankine cycle"""

'''온도, 압력의 array 설정 변화'''
T_steam = np.array(np.linspace(300,500,10))
"""단위 °C"""
P_boiler_HP = np.array(np.linspace(4,6,10))*10
P_boiler_LP = np.array(np.linspace(1.4,2.4,10))*10
"""bar"""
P_condenser = np.array(np.linspace(75,15,10))*0.01
"""kPa -> bar"""

"""해석"""
Efficiency = []
for i in range(len(T_steam)):
    for j in range(len(P_boiler_HP)):
        for k in range(len(P_boiler_LP)):
            for w in range(len(P_condenser)):
                """state_1 condenser to pump"""
                v_1 = s_t.vL_p(P_condenser[w])
                h_1 = s_t.hL_p(P_condenser[w])
                s_1 = s_t.sL_p(P_condenser[w])
                T_1 = s_t.t_ps(P_condenser[w],s_1)
                """state_2 pump to boiler"""
                W_pump_in = v_1*(P_boiler_HP[j]-P_condenser[w])*1000
                h_2 = h_1 +W_pump_in
                s_2 = s_t.s_ph(P_boiler_HP[j],h_2)
                T_2 = s_t.t_ps(P_boiler_HP[j],s_2)

                """state_3 boiler to High pressure Turbine"""
                h_3 = s_t.h_pt(P_boiler_HP[j],T_steam[i])
                s_3 = s_t.s_pt(P_boiler_HP[j],T_steam[i])
                T_3 = s_t.t_ps(P_boiler_HP[j],s_3)

                """state_4 HP turbine to regenerator"""
                s_4 = s_3
                h_4 = s_t.h_ps(P_boiler_LP[k],s_4)
                T_4 = s_t.t_ps(P_boiler_LP[k],s_4)
                """state_5 regenerator to LP turbine"""
                h_5 = s_t.h_pt(P_boiler_LP[k],T_steam[i])
                s_5 = s_t.s_pt(P_boiler_LP[k],T_steam[i])
                T_5 = s_t.t_ps(P_boiler_LP[k],s_5)
                """state_6 LP turbine to condenser"""
                s_6 = s_5
                quality = s_t.x_ps(P_condenser[w],s_6)
                h_6 = s_t.h_px(P_condenser[w],quality)
                T_6 = s_t.t_ps(P_condenser[w],s_6)
                """efficiency"""
                Q_in = (h_3 - h_2) + (h_5 - h_4)
                Q_out = h_6 - h_1

                W_net = Q_in - Q_out
                E = W_net/Q_in *100

                Efficiency.append([i,j,k,w,E,T_1,T_2,T_3,T_4,
                                 T_5,T_6,s_1,s_2,s_3,s_4,s_5,s_6])

"""max 열효율 인덱스 찾기"""
Efficiency_variable = np.array(Efficiency)
Thermal_Efficiency = []
for i in range(len(Efficiency_variable)):
    Thermal_Efficiency.append(Efficiency_variable[i][4])

Max_E_index = np.argmax(Thermal_Efficiency)
i,j,k,w = Efficiency_variable[Max_E_index][0:4]

E_max = Efficiency_variable[Max_E_index][4].round(2)
"""bar => MPa, kPa"""
T_max_steam = T_steam[int(i)]
P_max_HP = P_boiler_HP[int(j)]*0.1
P_max_LP = P_boiler_LP[int(k)]*0.1
P_min_con = P_condenser[int(w)]*100

print('최고의 열 효율 {}%, at T = {}°C, P_HP = {}MPa,'
      'P_LP = {}MPa, P_con = {}kPa'.format(E_max,T_max_steam,
                                     P_max_HP,P_max_LP,P_min_con))

"""그림 7개의 임의의 과정 선택 T_S 선도"""
import matplotlib.pyplot as plt

def data (index):
    x = []
    y = []
    for i in range(6):
        x.append(Efficiency_variable[index][i + 11] * 10)
        y.append(Efficiency_variable[index][i + 5])
    x.append(Efficiency_variable[index][11]*10)
    y.append(Efficiency_variable[index][5])
    """처음과 끝점 연결을 위해 마지막에 한번더 append"""
    return x,y

"""ploting을 위한 data 생성(s,T)"""
index = [0,1200,3000,6200,8000,9000,Max_E_index]
graph_data = []
for i in range(len(index)):
    graph_data.append(data(index[i]))

"""plotting후 비교를 위한 parameter설정"""
color = ['white','green','blue','yellow','black','magenta','red']
linestyle = ['solid','dotted','dashdot','dashdot','dashed','dashed','solid']
legend = ['1','2','3','4','5','6','MAX']

"""Plotting"""
plt.figure(figsize = (12,8))
plt.xlim([0,110])
plt.ylim([0, 700])
plt.xlabel('Entropy (kJ/kg°C)')
plt.ylabel('Temperature (°C)')
plt.text(90,-80,'Entropy X10 scaling for visual')
ax = plt.gca()
ax.set_facecolor('grey')
plt.grid(True)
for i in range(len(color)):
    x,y = graph_data[i]
    plt.scatter(x,y,color = color[i])
    plt.plot(x,y,linestyle = linestyle[i], color = color[i],lw = 3,label = legend[i])
    plt.legend()
plt.show()

"""Max efficiency => if ideal Turbine, Pump 가 아닐 경우의 3D pot"""
Pump_Efficiency = np.linspace(70,100,10)*0.01
Turbine_Efficiency = np.linspace(100,70,20)*0.01

i,j,k,w = Efficiency_variable[Max_E_index][0:4]
T = T_steam[int(i)]
P_HP = P_boiler_HP[int(j)]
P_LP = P_boiler_LP[int(k)]
P_con = P_condenser[int(w)]
# print(T,P_HP,P_LP, P_con)

Efficiency_actual = []
for a in range(len(Pump_Efficiency)):
    for b in range(len(Turbine_Efficiency)):
        """state_1 condenser to pump"""
        v_1 = s_t.vL_p(P_con)
        h_1 = s_t.hL_p(P_con)
        """state_2 pump to boiler"""
        W_pump_in = v_1 * (P_HP - P_con) * 1000
        h_2s = h_1 + W_pump_in
        h_2a = h_1 + (h_2s - h_1)/Pump_Efficiency[a]

        """state_3 boiler to High pressure Turbine"""
        h_3 = s_t.h_pt(P_HP, T)
        s_3 = s_t.s_pt(P_HP, T)

        """state_4 HP turbine to regenerator"""
        s_4 = s_3
        h_4s = s_t.h_ps(P_LP, s_4)
        h_4a = h_3 - Turbine_Efficiency[b]*(h_3 - h_4s)
        """state_5 regenerator to LP turbine"""
        h_5 = s_t.h_pt(P_LP, T)
        s_5 = s_t.s_pt(P_LP, T)

        """state_6 LP turbine to condenser"""
        s_6 = s_5
        quality = s_t.x_ps(P_con, s_6)
        h_6s = s_t.h_px(P_con, quality)

        h_6a = h_5 - Turbine_Efficiency[b]*(h_5 - h_6s)
        """efficiency"""
        Q_in = (h_3 - h_2a) + (h_5 - h_4a)
        Q_out = h_6a - h_1

        W_net = Q_in - Q_out
        E = W_net / Q_in * 100
        Efficiency_actual.append((a,b,E))

Efficiency_actual = np.array(Efficiency_actual)

"""plotting"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(69,101)
ax.set_ylim(69,101)
ax.set_zlim(10,50)
ax.set_zlabel('Efficiency X 1000')
ax.set_xlabel('Pump efficiency')
ax.set_ylabel('Turbine efficiency')

x_data = []
y_data = []
z_data = []
for i in range(len(Efficiency_actual)):
    x,y,z = Efficiency_actual[i]
    x_data.append(Pump_Efficiency[int(x)])
    y_data.append(Turbine_Efficiency[int(y)])
    z_data.append(z)

"""Data scaling"""
x_data = np.array(x_data)*100
y_data = np.array(y_data)*100
z_data = np.array(z_data)


# x_m, y_m = np.meshgrid(x_data, y_data)
ax.scatter(x_data,y_data,np.exp(z_data*0.1))
ax.text(100,100,np.exp(np.max(z_data)*0.1),
        'Max',color = 'red')
ax.text(70,70,np.exp(np.min(z_data)*0.1),'Min',color = 'blue')
# fig.colorbar(surf,shrink  = 0.5, aspect = 5)

plt.show()


"""각 온도와 압력의 데이터를 이용해 Sklearn의 Regression 모델을 이용해
    학습후 random한 온도와 압력에서의 열효율 에측값을 얻기 위한 모델링 과정"""
import pandas as pd
"""Data 생성"""
columns = ['T','P_HP','P_LP','P_cond','E']
temp = []
HP_pres = []
LP_pres = []
cond_pres = []

for i in range (len(Efficiency_variable)):
    a,b,c,d = Efficiency_variable[i][0:4]
    temp.append(T_steam[int(a)])
    HP_pres.append(P_boiler_HP[int(b)])
    LP_pres.append(P_boiler_LP[int(c)])
    cond_pres.append(P_condenser[int(d)])

data = pd.DataFrame(data = zip(temp,HP_pres,LP_pres,cond_pres,Thermal_Efficiency),
                    columns = columns)
print(data.head(8))

"""modeling with SKlearn"""
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(data.iloc[:,0:4],data.iloc[:,4],
                                                   random_state=42)
print(X_test)
print(y_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))
print(model.coef_)

A_cycle = [322 , 4.32*10 , 1.48*10 ,63*0.01]
B_cycle = [388.888889  ,55.555556  ,18.444444 , 0.616667]
C_cycle = [389 , 57, 20, 0.64]
# 388.888889  55.555556  18.444444  0.616667 ,30.169509
Sample  = np.array([A_cycle,B_cycle,C_cycle])
Sample = scaler.transform(Sample)
prediction_E = model.predict(Sample)
print(prediction_E)

