#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 12            #
# Due 5/8/21        #
#                   #
#####################

import numpy as np
import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack
import control as con

from numpy import sin, cos, pi, arange
from numpy.random import randint

#%% Noisy Signal Creation

fs = 1e6
Ts = 1/fs
t_end = 50e-3

t = arange(0, t_end-Ts, Ts)

f1 = 1.8e3
f2 = 1.9e3
f3 = 2e3
f4 = 1.85e3
f5 = 1.87e3
f6 = 1.94e3
f7 = 1.92e3

info_signal = 2.5*cos(2*pi*f1*t) + 1.75*cos(2*pi*f2*t) + 2*cos(2*pi*f3*t) + 2*cos(2*pi*f4*t) + 1*cos(2*pi*f5*t) + 1*cos(2*pi*f6*t) + 1.5*cos(2*pi*f7*t)

N = 25
my_sum = 0

for i in range(N + 1):
    noise_amp     = 0.075*randint(-10,10,size=(1,1))
    noise_freq    = randint(-1e6,1e6,size=(1,1))
    noise_signal  = my_sum + noise_amp * cos(2*pi*noise_freq*t)
    my_sum = noise_signal

f6 = 50e3
f7 = 49.9e3
f8 = 51e3

pwr_supply_noise = 1.5*sin(2*pi*f6*t) + 1.25*sin(2*pi*f7*t) + 1*sin(2*pi*f8*t)

f9 = 60

low_freq_noise = 1.5*sin(2*pi*f9*t)

total_signal = info_signal + noise_signal + pwr_supply_noise + low_freq_noise
total_signal = total_signal.reshape(total_signal.size)

plt.figure(figsize = (12,8))
plt.subplot(3,1,1)
plt.title('Main Noise Sources')
plt.plot(t,info_signal)
plt.ylabel('Position Signal')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t, info_signal + pwr_supply_noise)
plt.ylabel('Low Frequency Vibration')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, total_signal)
plt.ylabel('Switching Amplifier')
plt.grid()
plt.show()

df = pd.DataFrame({'0':t,'1':total_signal})

df.to_csv('NoisySignal.csv')

#%% Example Code

df = pd.read_csv('NoisySignal.csv')
t = df['0'].values
sensor_sig = df['1'].values

plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noise Input Signal')
plt.xlabel('Time[s]')
plt.ylabel('Amplitude [v]')
plt.show()

#%% Task 1

# Postion Measurement Information
# y = 200*np.cos(2*np.pi*100*t)

def my_fft(x, fs):
    N = len(x) 
    
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 
    freq = np.arange(-N/2, N/2) * fs/N 

    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)

    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0

    return freq, X_mag, X_phi
 
def make_stem(ax, x, y, color='k', style = 'solid', label=' ', linewidths=2/5,**kwargs):
     ax.axhline(x [0], x [-1], 0, color ='r')
     ax.vlines (x, 0 ,y , color=color , linestyles = style , label = label , linewidths=
     linewidths )
     ax.set_ylim ([1.05* y.min() , 1.05* y.max()])

# Before Filter

# x = total_signal
# freq, X_mag, X_phi = my_fft(x, fs)

# fig, ax= plt.subplots(figsize=(10,7))
# make_stem(ax, freq, X_mag)

fs=1e6
freq, X_mag, X_phi = my_fft(sensor_sig, fs)


fig,(ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize = (10,20))

plt.subplot(ax1)
plt.title('Before Filter Spectrum of Signals')
make_stem(ax1, freq, X_mag, linewidths = 2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1,100000])

plt.subplot(ax2)
make_stem(ax2,freq,X_mag,linewidths=2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([10,1000])

plt.subplot(ax3)
make_stem(ax3,freq,X_mag,linewidths = 2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1800,2200])

plt.subplot(ax4)
make_stem(ax4,freq,X_mag,linewidths = 2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([10000,100000])
plt.xlabel('Frequency(Hz)')
plt.tight_layout() 
plt.show ()


# plt.xlim(0,100)
# plt.ylim(0,1.3)

#plt.xscale('log')
#plt.title('Total Signal')
#plt.xlabel('freqency(Hz)')
#plt.ylabel('Magnitude')
#plt.xlim(10, 1e5)
#plt.show()

#%% Task 2

# Chosen values for circuit design 
C = 105e-9 
R = 1050 
L = 66e-3 

# Hm = (W/(R*C))/np.sqrt(W**4+(((1/(R*C))**2)-2/(L*C))*W**2+(1/(L*C))**2)
# Hp = 90-np.arctan((W/(R*C))/(1/(L*C)-(W**2)))*180/(np.pi)
#
#for i in range(len(Hp)):
#    if (Hp[i] > 90):
#        Hp[i] = Hp[i] - 180
       
#plt.figure(figsize = (7, 5))
#plt.subplot(2, 1, 1)
#plt.semilogx(W, 20*np.log10(Hm))
#plt.ylabel('H_mag(jw)')
#plt.title('Figure 1: H(jw)')
#plt.grid()

#plt.subplot(2, 1, 2)
#plt.semilogx(W, Hp)
#plt.ylabel('H_phase(jw)')
#plt.xlabel('w(rad/s)')
#plt.grid()

den = [1, 1/(R*C), 1/(L*C)]
num = [1/(R*C), 0]

r, p ,k = sig.residue(num,den)
print("R = ", r, "P = ", p)

steps = 100
W = np.arange(1, 10000000 + steps, steps)

func = sig.TransferFunction(num, den)
w, mag, phase = sig.bode(func, W)

#%% Task 3 

plt.figure(figsize = (10,30))
plt.subplot(8,1,1)
plt.semilogx(w/(2*np.pi), mag)   
plt.ylabel('Gain(dB)')
plt.grid()
plt.title('Bode Gain')

# Postion Plot

plt.subplot(8,1,2)
plt.semilogx(w/(2*np.pi), mag)   
plt.ylabel('Gain(dB)')
plt.xlim([1800,2200])
plt.ylim([-3,0.8])
plt.grid()
plt.title('Position')

# Low Frequency Plot

plt.subplot(8,1,3)
plt.semilogx(w/(2*np.pi), mag)   
plt.ylabel('Gain(dB)')
plt.xlim([30,100])
plt.ylim([-50,0.8])
plt.grid()
plt.title('Low Frequency')

# High Frequency Plot

plt.subplot(8,1,4)
plt.semilogx(w/(2*np.pi), mag)   
plt.ylabel('Gain(dB)')
plt.xlim([100000,1000000])
plt.ylim([-100,0])
plt.grid()
plt.title('High Frequency')

# Switching Amp Frequency Plot

plt.subplot(8,1,5)
plt.semilogx(w/(2*np.pi), mag)   
plt.ylabel('Gain(dB)')
plt.xlabel('Frequency(Hz)')
plt.xlim([10000,100000])
plt.ylim([-50,0.8])
plt.grid()
plt.title('Switching Amp')
plt.tight_layout()


#%% Task 4 

steps = 1e-6
t = np.arange(0, 5e-2+steps, steps)

numZ, denZ = sig.bilinear(num, den, 1/steps)

y = sig.lfilter(numZ, denZ, sensor_sig)
plt.figure(figsize=(10,10))

plt.plot(t[0:50000], y)
plt.xlabel('Time')
plt.ylabel('Output') 
plt.title('After Filter Design')
plt.grid()
plt.show()


# FFT after filter

freq, X_mag, X_phi = my_fft(y,fs)

fig,(ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(10,20))
plt.subplot(ax1)
plt.title('After Filter Plots')
make_stem(ax1,freq,X_mag,linewidths=2.5)
plt.xscale('log')
plt.xlim([1,100000])
plt.ylim([0,1.8])

plt.subplot(ax2)
make_stem(ax2,freq,X_mag,linewidths=2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([10,1000])

plt.subplot(ax3)
make_stem(ax3,freq,X_mag,linewidths=2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([1800,2200])

plt.subplot(ax4)
make_stem(ax4,freq,X_mag,linewidths=2.5)
plt.xscale('log')
plt.ylim([0,1.8])
plt.xlim([10000,100000])

plt.xlabel('Frequency(Hz)')
plt.show()
















































