#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 6             #
# 3/2/21            #
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
t = np.arange(0, 2 + steps, steps)

# Hand calculated from Prelab
def h_1(t):
    y = ((1/2) + np.exp(-6*t) - ((1/2)*np.exp(-4*t))) * step_func(t) 
    return y

y1 = h_1(t)

plt.figure(figsize=(10,7))
plt.plot(t, y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response y(t)')
plt.show()

#%% Part 1 - Task 2

num1 = [1,6,12]
den1 = [1,10,24]

tout, yout = sig.step((num1, den1), T = t)

plt.figure(figsize=(10,7))
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response H(s)')
plt.show()

#%% Part 1 - Task 3

num2 = [1,6,12]
den2 = [1,10,24,0]

y2 = sig.residue(num2,den2)

print(y2[0])
print(y2[1])
print(y2[2])


#%% Part 2 - Task 1

steps = 1e-5

t = np.arange(0, 4.5+steps, steps)

num3 = [25250]
den3 = [1,18,218,2036,9085,25250,0]

r,p,_ = sig.residue(num3,den3)

print('r=', r, "\n" 'p=', p)

#%% Part 2 - Task 2

y_cos = 0
for i in range(len(r)):
    phasek = np.angle(r[i])
    magk = np.abs(r[i])
    w = np.imag(p[i])
    a = np.real(p[i])   
    y_cos += magk*np.exp(a*t)*np.cos(w*t + phasek)*step_func(t)
    

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(t, y_cos)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response y(t)')

#%% Part 2 - Task 3

num4 = [25250]
den4 = [1,18,218,2036,9085,25250]

t4, y4 = sig.step((num4, den4), T = t)

plt.subplot(2,1,2)
plt.plot(t4, y4)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('step response H(s)')
plt.tight_layout()
plt.show()



