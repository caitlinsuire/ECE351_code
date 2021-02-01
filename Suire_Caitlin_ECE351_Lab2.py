#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 2             #
# Due 2/4/21        #
#                   #
#####################


#%% Part 1 - Task 1


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})  # set font size


steps = 1e-2     # Step size
 

t = np.arange(0, 10 + steps, steps)


print ('Number of elements : len(t) =', len(t), '\nFirst Element: t[0] =', t[0], '\nLastElement: t[len(t) -1] =', t[len(t) -1])


#%% Part 1 - Task 2

def func1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if i < (len(t) + 1)/3:
            y[i] = t[i]**2
        else:
           y[i] = np.sin(5*t[i])+2
    return y 

y = np.cos(t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('Function 1')
plt.title('Part 1 - Cosine Function')
plt.show()


#%% Part 2 - Task 1 

#        y(t) = r(t) - r(t-3) + 5u(t-3) -2u(t-6) -2r(t-6) 


#%% Part 2 - Task 2 

def ramp(t):
    y = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else: 
            y[i] = 0
    return y

steps = 1e-3
t = np.arange(-1, 1+ steps, steps)
    
plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t, ramp(t))
plt.grid()
plt.ylabel('r(t)')
plt.title('Part 2 - Ramp Function')
plt.ylim([-0.5, 1.5])
plt.show()


def step(t):
    y = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

steps = 1e-3
t = np.arange(-1,1+steps, steps)

plt.figure(figsize = (10,7))
plt.subplot(2,1,2)
plt.plot(t, step(t))
plt.grid()
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Part 2 - Step Function')
plt.ylim([-0.5, 1.5])
plt.show()


#%% Part 2 - Task 3 

def func2(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))


steps = 1e-3

t = np.arange(-5, 10+steps, steps)
y=func2(t)

plt.figure(figsize = (10,7))
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Part 3 - Plotting Function')
plt.ylim([-2,10])
plt.xsticks([-5,0,1,2,3,4,5,6,7,8,9,10])
plt.ysticks([-5,0,1,2,3,4,5,6,7,8,9,10])

plt.show()



#%% Part 3 - Task 1 


steps = 1e-3

t = np.arange(-10, 8+steps, steps)
y = func2(-t)

plt.figure(figsize=(10,7))
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Time Reversal Plot')
plt.ylim([-2,10])
plt.xticks([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,5])
plt.yticks([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
plt.show()



#%% Part 3 - Task 2 


steps = 1e-3

t = np.arange(-20,20+steps, steps)
y = func2(t-4)
y1 = func2(-t-4)

plt.figure(figsize=(10,7))
plt.tight_layout()
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Time Shift Operations f(t-4)')
plt.ylim([-2,10])
plt.xticks([0,2,4,7,10,12,14])
plt.yticks([0,1,2,3,4,5,6,7,8,9,10])

plt.subplot(2,1,2)
plt.plot(t,y1)
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Time Shift Operations f(-t-4)')
plt.ylim([-2,10])
plt.xticks([-16,-14,-12,-10,-7,-4,0])
plt.yticks([0,1,2,3,4,5,6,7,8,9,10])

plt.show()



#%% Part 3 - Task 3 


steps = 1e-3

t = np.arange(-5, 20+steps, steps)
y1 = func2(t/2)
y2 = func2(t*2)

plt.figure(figsize=(10,7))
plt.plot(t, y1, label = 'f(t/2)')
plt.plot(t, y2, label = 'f(t*2)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Time Scaling Operations')
plt.legend()
plt.grid()
plt.ylim([-2,10])
plt.show()



#%% Part 3 - Task 5 


steps = 1e-2

t = np.arange(-5, 10+steps, steps)
y = func2(t)

dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt

plt.figure(figsize=(10,7))
plt.plot(t, y, '--', label = 'y(t)')
plt.plot(t[range(len(dy))], dy[:,0], label = 'dy(t)/dt')
plt.xlabel('t')
plt.ylabel('y(t), dy(t)/dt')
plt.title('Derivative with Respect to Time')
plt.legend()
plt.grid()
plt.ylim([-2,10])
plt.show()







