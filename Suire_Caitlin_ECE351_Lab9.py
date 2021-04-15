#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 9             #
# 3/30/21           #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
import matplotlib.gridspec

#%% Part 1 - Task 1

def fft1(f, fs):
    N = len(x)                            
    X_fft = fft(x)                     
    X_fft_shifted = fftshift(X_fft)   
                                                
    freq = np.arange(-N/2 , N/2) * fs/N  
                                                       
    X_mag = np.abs(X_fft_shifted)/N                   
    X_phi = np.angle(X_fft_shifted)    
            
    return freq, X_mag, X_phi

def fft2(f, fs):
    N = len(x)                            
    X_fft = fft(x)                     
    X_fft_shifted = fftshift(X_fft)   
                                                
    freq = np.arange(-N/2 , N/2) * fs/N  
                                                       
    X_mag = np.abs(X_fft_shifted)/N                   
    X_phi = np.angle(X_fft_shifted)    
    
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10: 
            X_phi[i] = 0
            
    return freq, X_mag, X_phi

def fft_plot(t, freq, x, X_mag, X_phi):
    
    plt.figure(figsize = (10,7))
    plt.subplot(3,1,1)
    plt.plot(t,x)
    plt.grid()
    plt.xlabel('t[s]')
    plt.ylabel('x(t)')
    plt.title('FFT')
    
    plt.subplot(3,2,3)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.ylabel('|X(f)|')

    plt.subplot(3,2,4)
    plt.stem(freq, X_mag)
    plt.grid()
    plt.xlim([-2,2])
    
    plt.subplot(3,2,5)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.ylabel('/_ X(f)')
    plt.xlabel('f[Hz]')
    
    plt.subplot(3,2,6)
    plt.stem(freq, X_phi)
    plt.grid()
    plt.xlim([-2,2])
    plt.xlabel('f[Hz]')
    
    plt.tight_layout()
    plt.show()
    
    return 0


fs = 100
steps = 1/fs
t = np.arange(0,2,steps)

x = np.cos(2*np.pi*t)
f, mag, phi = fft1(x,fs)
fft_plot(t, f, x, mag, phi)

#%% Task 2
           
x2 = 5*np.sin(2*np.pi*t)
f, mag, phi = fft1(x2, fs)
fft_plot(t, f, x2, mag, phi)

#%% Task 3

x3 = (2*np.cos(2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2
f, mag, phi = fft1(x3, fs)
fft_plot(t, f, x3, mag, phi)

#%% Task 4a

f, mag, phi = fft2(x, fs)
fft_plot(t, f, x, mag, phi)

#%% Task 4b

f, mag, phi = fft2(x2, fs)
fft_plot(t, f, x2, mag, phi)

#%% Task 4c

f, mag, phi = fft2(x3, fs)
fft_plot(t, f, x3, mag, phi)

#%% Task 5

t = np.arange(0, 16, steps)
T = 8
x5 = 0

for k in np.arange(1, 15+1):
    b = 2/(k*np.pi)*(1-np.cos(k*np.pi))
    x = b*np.sin(k*(2*np.pi/T)*t)
    x5 += x
    
f, mag, phi = fft2(x5, fs)
fft_plot(t, f, x5, mag, phi)
