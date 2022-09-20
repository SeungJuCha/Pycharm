import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

"""데이터 로딩"""
file = "C:\\Users\\seungju\\PycharmProjects\\pythonProject1\\venv1\\SineWave_SquareWave.csv"

data = pd.read_csv(file,header = 0)
data = data.drop(index = 0,axis = 0)
Data_frame = pd.DataFrame(data = data)

sin_x,sin_y = Data_frame['Time'],Data_frame['Voltage']
squre_x,squre_y =Data_frame['Time.1'],Data_frame['Voltage.1']
#x는 time y는 진폭

sin_x = np.array(sin_x)
sin_y = np.array(sin_y)
squre_x = np.array(squre_x)
squre_y = np.array(squre_y)

idx = []
for i in range(len(sin_y)):
    if sin_y[i]>= 5.0:
        idx.append(i)

Time = []
for a in range(len(idx)):
    t = idx[a]
    Time.append(sin_x[t])
print(Time)
print(idx)
#위 비교를 통한 아래 주기값
print('Time:', 0.012636208-0.002559232) # 첫번쨰 주기

"""sinWave"""

def FFT(Freq,x,y):
    Fs = Freq
    # Calculate FFT ....................
    n = len(y)
    NFFT = n
    k = np.arange(NFFT)
    f0 = k * Fs / NFFT  # double sides frequency range
    f0 = f0[range(math.trunc(NFFT / 2))]  # single sied frequency range

    Y = np.fft.fft(y) / NFFT  # fft computing and normaliation
    Y = Y[range(math.trunc(NFFT / 2))]  # single sied frequency range
    amplitude_Hz = 2 * abs(Y)
    phase_ang = np.angle(Y) * 180 / np.pi

    print(amplitude_Hz.max())
    f_Hz = []
    for i in range(len(amplitude_Hz)):
        f_Hz.append((f0[i],amplitude_Hz[i]))

    sort_f_Hz = sorted(f_Hz,key = lambda f_Hz: f_Hz[1],reverse = True)
    print(sort_f_Hz[0:4])
    # print(amp_max, f_point)


    # figure 1 ..................................
    plt.figure(num=2, dpi=100, facecolor='white')
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.subplot(3, 1, 1)

    plt.plot(x, y, 'r')
    plt.title('Signal FFT analysis')
    plt.xlabel('time($sec$)')
    plt.ylabel('y')

    # Amplitude ....
    # plt.figure(num=2,dpi=100,facecolor='white')
    plt.subplot(3, 1, 2)

    # Plot single-sided amplitude spectrum.

    plt.plot(f0, amplitude_Hz, 'r')  # 2* ???
    plt.xticks(np.arange(0, 500, 20))
    plt.xlim(0, 60)
    plt.ylim(0, 10)
    # plt.title('Single-Sided Amplitude Spectrum of y(t)')
    plt.xlabel('frequency($Hz$)')
    plt.ylabel('amplitude')
    plt.grid()

    # Phase ....
    # plt.figure(num=2,dpi=100,facecolor='white')
    plt.subplot(3, 1, 3)
    plt.plot(f0, phase_ang, 'r')  # 2* ???
    plt.xlim(0, 60)
    plt.ylim(-180, 180)
    # plt.title('Single-Sided Phase Spectrum of y(t)')
    plt.xlabel('frequency($Hz$)')
    plt.ylabel('phase($deg.$)')
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.yticks([-180, -90, 0, 90, 180])
    plt.grid()
    plt.show()

sin_FFT = FFT(100,sin_x,sin_y)
print(sin_FFT)

print(FFT(100,squre_x,squre_y))

