#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 7             #
# 3/9/21            #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Part 1 - Task 1

# G(s) = (s+9)/((s-8)(s+2)(s+4))

# A(s) = (s+4)/((s+3)(s+1))

# B(s) = (s+12)(s+14)

#%% Part 1 - Task 2

# G(s)
numg = [1,9]
deng = sig.convolve([1,-6,-16],[1,4])
Z1, P1,_ = sig.tf2zpk(numg, deng)
print('Z of G: ', Z1, '  P of G: ', P1)

# A(s)
numa = [1,4]
dena = [1,4,3]
Z2, P2,_ = sig.tf2zpk(numa,dena)
print('Z of A: ', Z2, '  P of A: ', P2)

# B(s)
numb = [1,26,168]
root = np.roots(numb)
print('Z of B: ', root)

#%% Part 1 - Task 3

open_loop_num = sig.convolve(numg, numa)
open_loop_den = sig.convolve(deng, dena)
print('Open Loop Num: ', open_loop_num, '  Open Loop Den: ', open_loop_den)

#%% Part 1 - Task 4

# The open-loop response is not stable since
# it has a pole (s - 8) in the right side of the real-imaginary plane

#%% Part 1 - Task 5

numOpen = sig.convolve(numa, numg)
denOpen = sig.convolve(dena, deng)
tout,yout = sig.step((numOpen, denOpen))

plt.figure(figsize = (10,7))
plt.plot(tout, yout)
plt.grid()
plt.title('Step Response of Open-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

#%% Part 1 - Task 6

# Yes, supports answer
# Not stable because there is a pole in the negative s-plane

#%% Part 2 - Task 1

# Screenshot on Phone - Type in LaTex

#%% Part 2 - Task 2

numTotal = sig.convolve(numa, numg)
denTotal = sig.convolve(deng + sig.convolve(numb, numg), dena)
zTotal, pTotal, _ = sig.tf2zpk(numTotal, denTotal)

print('Closed-Loop Num: ', numTotal, '\nClosed-Loop Den: ', denTotal)
print('Closed-Loop Z: ', zTotal, '\nClosed-Loop P: ', np.round(pTotal,2))

#%% Part 2 - Task 3

# Stable because poles are in the positive s-plane 

#%% Part 2 - Task 4

tout, yout = sig.step((numTotal, denTotal))

plt.figure(figsize = (10,7))
plt.plot(tout, yout)
plt.grid()
plt.title('Step Response of a Closed-Loop Transfer Function')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()

#%% Part 3 - Task 5

# Yes, supports answer
# Stable because poles are all in the positive s-plane 



