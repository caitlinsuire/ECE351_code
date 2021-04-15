#####################
#                   #
# Caitlin Suire     #
# ECE 351-53        #
# Lab 11            #
# 4/13/21           #
#                   #
#####################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from matplotlib import patches  

def step_func(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

steps = 1e-5
t = np.arange(0, 2+steps, steps)

#%% Part 1 - Task 3

num = [2, -40]
den = [1, -10, 16]

r1, p1, _ = sig.residuez(num, den)
print('r= ', r1, 'p= ', p1)

#%% Zplane Code Given

# -*- coding: utf-8 -*-
"""
@author: Phillip Hagen

Description:zplane()


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

#%% Zplane function
"""

#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.
#
#
#
# Modified by Drew Owens in Fall 2018 for use in the University of Idaho's 
# Department of Electrical and Computer Engineering Signals and Systems I Lab
# (ECE 351)
#
# Modified by Morteza Soltani in Spring 2019 for use in the ECE 351 of the U of
# I.
#
# Modified by Phillip Hagen in Fall 2019 for use in the University of Idaho's  
# Department of Electrical and Computer Engineering Signals and Systems I Lab 
# (ECE 351)
    
def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

#    import numpy as np
#    import matplotlib.pyplot as plt
#    from matplotlib import patches    
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

#%% Part 1 - Task 4

z, p, k = zplane(num, den)
print('z= ', z, 'p= ', p, 'k= ', k)

#%% Part 1 - Task 5

num = [2, -40]
den = [1, -10, 16]

w, h = sig.freqz(num, den, whole = True)

w = w/np.pi
mag = np.abs(h)
mag = 20*np.log10(mag)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.title('Magnitude Response')
plt.plot(w,mag)
plt.grid()

plt.subplot(2,1,2)
plt.title('Phase Response')
plt.plot(w, np.angle(h))
plt.grid()
plt.show()

