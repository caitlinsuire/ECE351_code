#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 3             #
# Due 2/4/21        #
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
    
def ramp_func(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y
    
def f_1(t):
    y = step_func(t-2) - step_func(t-9)
    return y

def f_2(t):
    y = np.exp(-t)*step_func(t)
    return y

def f_3(t):
    y = ramp_func(t-2)*(step_func(t-2)-step_func(t-3)) + ramp_func(4-t)*(step_func(t-3)-step_func(t-4))
    return y
 
# Part 1 - Task 2 
  
steps = 1e-3
t = np.arange(-1, 6+steps, steps)

y1 = f_1(t)
y2 = f_2(t)
y3 = f_3(t)
   
plt.figure(figsize=(10,7))
plt.plot(t, y1, label = 'f_1(t)')
plt.plot(t, y2, label = 'f_2(t)')
plt.plot(t, y3, label = 'f_3(t)')
plt.xlabel('t')
plt.title('Function Plots')
plt.legend()
plt.grid()
plt.show()


#%% Part 2 - Task 1

def conv_func(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2 -1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1 -1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if(i - j + 1 > 0):
                try:
                    result[i] += f1Extended[j]*f2Extended[i - j + 1]
                except:
                        print(i,j)
    return result

steps = 1e-2
t = np.arange(0, 20+steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN -1], steps)

f1 = f_1(t)
f2 = f_2(t)
f3 = f_3(t)


# Part 2 - Task 2

conv12 = conv_func(f1, f2)*steps
conv12Check = sig.convolve(f1, f2)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, conv12, label = 'User-Defined Convolution')
plt.plot(tExtended, conv12Check, '--', label = 'Built-In Convolution')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv12')
plt.xticks(np.arange(0,21,1))
plt.show()

#%% Part 2 - Task 3

def conv_func(f2, f3):
    Nf2 = len(f2)
    Nf3 = len(f3)
    f2Extended = np.append(f2, np.zeros((1, Nf3 -1)))
    f3Extended = np.append(f3, np.zeros((1, Nf2 -1)))
    result = np.zeros(f2Extended.shape)
    
    for i in range(Nf3 + Nf2 - 2):
        result[i] = 0
        for j in range(Nf2):
            if(i - j + 1 > 0):
                try:
                    result[i] += f2Extended[j]*f3Extended[i - j + 1]
                except:
                        print(i,j)
    return result

steps = 1e-2
t = np.arange(0, 20+steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN -1], steps)

f1 = f_1(t)
f2 = f_2(t)
f3 = f_3(t)

conv23 = conv_func(f2, f3)*steps
conv23Check = sig.convolve(f2, f3)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, conv23, label = 'User-Defined Convolution')
plt.plot(tExtended, conv23Check, '--', label = 'Built-In Convolution')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv23')
plt.xticks(np.arange(0,21,1))
plt.show()

#%% Part 2 - Task 4

def conv_func(f1, f3):
    Nf1 = len(f1)
    Nf3 = len(f3)
    f1Extended = np.append(f1, np.zeros((1, Nf3 -1)))
    f3Extended = np.append(f3, np.zeros((1, Nf1 -1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf3 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if(i - j + 1 > 0):
                try:
                    result[i] += f1Extended[j]*f3Extended[i - j + 1]
                except:
                        print(i,j)
    return result

steps = 1e-2
t = np.arange(0, 20+steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN -1], steps)

f1 = f_1(t)
f2 = f_2(t)
f3 = f_3(t)

conv13 = conv_func(f1, f3)*steps
conv13Check = sig.convolve(f1, f3)*steps

plt.figure(figsize = (10,7))
plt.plot(tExtended, conv13, label = 'User-Defined Convolution')
plt.plot(tExtended, conv13Check, '--', label = 'Built-In Convolution')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv13')
plt.xticks(np.arange(0,21,1))
plt.show()




    
    
    
    
    