#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 8             #
# 3/23/21           #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Part 1 - Task 1

steps = 1e-2
t = np.arange(0,20+steps,steps)
T = 8

a = np.zeros((1501,1))
for k in np.arange(1,1501):
    a[k] = 0
    
b = np.zeros((1501,1))
for k in np.arange(1,1501):
    b[k] = 2/(k*np.pi)*(1-np.cos(k*np.pi))
    
print('a0 = ', a[0], 'a1 = ', a[1])
print('b1 = ', b[1], 'b2 = ', b[2], 'b3 = ', b[3])


#%% Part 1 - Task 2

# N = 1    
total = 0
N = 1
for k in np.arange(1, N+1):
    total = total + (b[k] * np.sin(k*(2*t*np.pi / T)))
    
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 1')

# N = 3
total = 0
N = 3
for k in np.arange(1,N+1):
    total = total + (b[k]*np.sin(k*(2*t*np.pi / T)))
    
plt.subplot(3,1,2)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 3')

# N = 15
total = 0
N = 15
for k in np.arange(1,N+1):
    total = total + (b[k]*np.sin(k*(2*t*np.pi / T)))
    
plt.subplot(3,1,3)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 15')
plt.tight_layout()
plt.show()

# N = 50
total = 0
N = 50
for k in np.arange(1,N+1):
    total = total + (b[k]*np.sin(k*(2*t*np.pi / T)))
    
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 50')

# N = 150
total = 0
N = 150
for k in np.arange(1,N+1):
    total = total + (b[k]*np.sin(k*(2*t*np.pi / T)))
    
plt.subplot(3,1,2)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 150')

# N = 1500
total = 0
N = 1500
for k in np.arange(1,N+1):
    total = total + (b[k]*np.sin(k*(2*t*np.pi / T)))
    
plt.subplot(3,1,3)
plt.plot(t,total)
plt.grid()
plt.title('Fourier Series Approximation')
plt.xlabel('t')
plt.ylabel('N = 1500')
plt.tight_layout()
plt.show()





