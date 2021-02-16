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

#%% Functions

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

#%% Part 1 - Task 1

def h_1(t):
    y = np.exp(2*t) * step_func(1-t)
    return y

def h_2(t):
    y = step_func(t-2)-step_func(t-6)
    return y

def h_3(t):
    y = np.cos(1.571*t)*step_func(t)
    return y

#%% Part 1 - Task 2

steps = 1e-3
t = np.arange(-10, 10+steps, steps)

y1 = h_1(t)
y2 = h_2(t)
y3 = h_3(t)

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(t, y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('y1(t)')
plt.title('h_1(t)', y=0.7, x=0.05)


plt.subplot(3,1,2)
plt.plot(t,y2)
plt.grid()
plt.xlabel('t')
plt.ylabel('y2(t)')
plt.title('h_2(t)', y=0.7, x=0.05)

plt.subplot(3,1,3)
plt.plot(t,y3)
plt.grid()
plt.xlabel('t')
plt.ylabel('y3(t)')
plt.title('h_3(t)', y=0.7, x=0.05)


plt.tight_layout()
plt.show()

#%% Part 2 - Task 1

# Convolution of F1

steps = 1e-3
A = len(t)
stepped = step_func(t)
tExtend = np.arange(2*t[0], 2*t[A-1]+steps, steps)

f1 = h_1(t)
f2 = h_2(t)
f3 = h_3(t)

conv1 = sig.convolve(f1, stepped)*steps

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(tExtend, conv1, label = 'built-in conv')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv of 1')
plt.xticks(np.arange(-20,20,5))

# Conv of F2

conv2 = sig.convolve(f2, stepped)*steps

plt.subplot(3,1,2)
plt.plot(tExtend, conv2, label = 'built-in conv')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv of 2')
plt.xticks(np.arange(-20,20,5))


# Conv of F3

conv3 = sig.convolve(f3, stepped)*steps


plt.subplot(3,1,3)
plt.plot(tExtend, conv3, label = 'built-in conv')
plt.grid()
plt.xlabel('t')
plt.ylabel('conv of 3')
plt.xticks(np.arange(-20,20,5))


plt.tight_layout()
plt.show()

#%% Part 2 - Task 2

# Hand Calculations

def h_4(t):
    y = (1/2)*np.exp(2*t)*step_func(1-t) + np.exp(2)*step_func(t-1)
    return y

def h_5(t):
    y = (t-2)*step_func(t-2) - (t-6)*step_func(t-6)
    return y

def h_6(t):
    y = (1/1.571)*np.sin(1.571*t)*step_func(t)
    return y

y4 = h_4(t)
y5 = h_5(t)
y6 = h_6(t)

plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(t, y4, label = 'h_4(t)')
plt.grid()
plt.xlabel('t')
plt.ylabel('h4(t)')
plt.title('h4(t)', y=0.7, x=0.05)


plt.subplot(3,1,2)
plt.plot(t, y5, label = 'h_5(t)')
plt.grid()
plt.xlabel('t')
plt.ylabel('h5(t)')
plt.title('h5(t)', y=0.7, x=0.05)


plt.subplot(3,1,3)
plt.plot(t, y6, label = 'h_6(t)')
plt.grid()
plt.xlabel('t')
plt.ylabel('h6(t)')
plt.title('h6(t)', y=0.7, x=0.05)

plt.show()

