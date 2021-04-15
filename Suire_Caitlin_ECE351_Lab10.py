#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 10            #
# 4/6/21            #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

#%% Part 1 - Task 1

def adjustHdeg(Hdeg):
    for i in range(len(Hdeg)):
        if Hdeg[i] > 90:
            Hdeg[i] = Hdeg[i] - 180
    return Hdeg

steps = 1e3
R = 1e3
L = 27e-3
C = 100e-9

omega = np.arange(1e3, 1e6 + steps, steps)

Hmag = (20*np.log10((omega/(R*C))/(np.sqrt(omega**4 + (1/(R*C)**2 - 2/(L*C))*omega**2 + (1/(L*C))**2))))

Hdeg = (np.pi/2 - np.arctan((1/(R*C)*omega)/(-omega**2 + 1/(L*C)))) * 180/np.pi
Hdeg = adjustHdeg(Hdeg)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(omega, Hmag)
plt.grid()
plt.ylabel('Magnitude in dB')
plt.title('Part 1 - Task 1')

plt.subplot(2,1,2)
plt.semilogx(omega, Hdeg)
plt.yticks([-90, -45, 0, 45, 90])
plt.ylim([-90,90])
plt.grid()
plt.ylabel('Phase in degrees')
plt.xlabel('Frequency in rad/s')
plt.show()

#%% Part 1 - Task 2

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

omega, Hmag, Hdeg = sig.bode((num, den), omega)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(omega, Hmag)
plt.grid()
plt.xlim([1e3, 1e6])
plt.title('Part 1 - Task 2')

plt.subplot(2,1,2)
plt.semilogx(omega, Hdeg)
plt.grid()
plt.xlim([1e3, 1e6])
plt.ylim([-90,90])
plt.xlabel('Frequency in rad/s')
plt.show()

#%% Part 1 - Task 3

sys = con.TransferFunction(num, den)
_ = con.bode(sys, omega, dB = True, Hz = True, deg = True, Plot = True)


#%% Part 2 - Task 1

fs = 50000*2*np.pi
steps = 1/fs
t = np.arange(0, .01 + steps, steps)
xt = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.plot(t, xt)
plt.grid()
plt.title('Part 2 - Task 1')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()

#%% Part 2 - Task 2

zxt, pxt = sig.bilinear(num, den, fs)

#%% Part 2 - Task 3

yt = sig.lfilter(zxt, pxt, xt)

#%% Part 2 - Task 4

plt.figure(figsize = (10,7))
plt.plot(t, yt)
plt.grid()
plt.title('Part 2 - Task 4')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()






