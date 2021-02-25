#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 4             #
# Due 2/16/21       #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Part 1 - Task 1

def step_func(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps)
R = 1e3
L = 27e-3
C = 100e-9

def sine_method(R, L, C, t):
    w = 0.5* np.sqrt((1/(R*C))**2 - 4*(np.sqrt(1/(L*C)))**2 + (0*1j))
    a = -1/(2*R*C)
    p = a + w
    g = (1/(R*C)) * p
    g_abs = np.abs(g)
    g_phase = np.angle(g)
    y1out = (g_abs / abs(w)) * np.exp(a*t)*np.sin(abs(w)*t + g_phase) * step_func(t)
    return y1out

#%% Part 1 - Task 2
num = [0, 1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

tout, yout = sig.impulse((num, den), T = t)


y1 = sine_method(R, L, C, t)

# Plots 
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t, y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('yout1')
plt.title('impulse hand-calculated')


plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t')
plt.ylabel('yout2')
plt.title('impulse actual')

plt.show()

#%% Part 2 - Task 1

tout2, yout2 = sig.step((num, den), T = t)

y1 = sine_method(R, L, C, t)
plt.figure(figsize=(10,7))
plt.plot(tout2, yout2)
plt.grid()
plt.xlabel('t')
plt.ylabel('yout3')
plt.title('impulse calculated pt2')
plt.show()




