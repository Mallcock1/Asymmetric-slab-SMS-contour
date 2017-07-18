# -*- coding: utf-8 -*-
"""
Created on Fri 7 July 15:16:32 2017

@author: Matthew
"""

import numpy as np
import scipy as sc
from scipy.optimize import newton
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

# SBS
# Define the sound speeds and alfven speeds.
c2 = 1.2
c0 = 1.
vA = 1.3
cT = sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))
R2 = 2.
#c1 = c2*np.sqrt(R2/R1)

mode_options = ['slow-kink-surf', 'slow-saus-surf', 'fast-kink-surf',
                'fast-saus-surf']

kink_mode_options = ['kink']
saus_mode_options = ['saus']

for mode in mode_options:
    if 'kink' in mode:
        kink_mode_options.append(mode)
    if 'saus' in mode:
        saus_mode_options.append(mode)

mode = mode_options[3]

#plot_variable = 'amp-ratio'
plot_variable = 'min-pert-shift'
#plot_variable = 'W'

print('Plotting ' + plot_variable + ' for ' + mode + ' mode')

###############################################################################

def m0(W):
    m0function = sc.sqrt((c0**2-W**2)*(vA**2-W**2)*((c0**2+vA**2)*(cT**2-W**2))**(-1))
    return m0function
    
def m1(W, R1):
    m1function = sc.sqrt(1-W**2*c2**(-2)*R1*R2**(-1))
    return m1function

def m2(W):
    m2function = sc.sqrt(1-W**2*c2**(-2))
    return m2function

error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
error_string_subscript = "subscript argument must be 1 or 2"

def disp_rel_sym(W, K, R1, mode, subscript):
    if subscript == 1:
        if mode in kink_mode_options:
            dispfunction = (vA**2 - W**2)*m1(W, R1)*sc.tanh(m0(W)*K) - R1*W**2*m0(W)
        elif mode in saus_mode_options:
            dispfunction = (vA**2 - W**2)*m1(W, R1) - R1*W**2*m0(W)*sc.tanh(m0(W)*K)
        else:
            print(error_string_kink_saus)
    elif subscript == 2:
        if mode in kink_mode_options:
            dispfunction = (vA**2 - W**2)*m2(W)*sc.tanh(m0(W)*K) - R1*W**2*m0(W)
        elif mode in saus_mode_options:
            dispfunction = (vA**2 - W**2)*m2(W) - R1*W**2*m0(W)*sc.tanh(m0(W)*K)
        else:
            print(error_string_kink_saus)
    else:
        print(error_string_subscript)
    return dispfunction
    
def disp_rel_asym(W, K, R1):
    return ((W**4 * m0(W)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W,R1) * m2(W) -
            0.5 * m0(W) * W**2 * (vA**2 - W**2) * (R2 * m1(W,R1) + R1 * m2(W)) *
            (sc.tanh(m0(W) * K) + (sc.tanh(m0(W) * K))**(-1))) /
            (vA**2 - W**2) * (c0**2 - W**2) * (cT**2 - W**2))
    
def amp_ratio(W, K, R1, mode):
    if mode in kink_mode_options:
        ampfunction = m2(W)*disp_rel_sym(W,K,R1,'saus',1) / (m1(W,R1)*disp_rel_sym(W,K,R1,'saus',2))
    elif mode in saus_mode_options:
        ampfunction = - m2(W)*disp_rel_sym(W,K,R1,'kink',1) / (m1(W,R1)*disp_rel_sym(W,K,R1,'kink',2))
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def min_pert_shift(W, K, R1, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W)) * sc.arctanh(- disp_rel_sym(W,K,R1,'saus',1) / disp_rel_sym(W,K,R1,'kink',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W)) * sc.arctanh(- disp_rel_sym(W,K,R1,'kink',1) / disp_rel_sym(W,K,R1,'saus',1))
    else:
        print(error_string_kink_saus)
    return shiftfunction

def cutoff(R1):
    c1 = c2*np.sqrt(R2/R1)
    if c2 < c1:
        cutoff = 0.5 * np.sqrt(((c0**2 + vA**2) * (cT**2 - c2**2)) / ((c0**2 - c2**2) * (vA**2 - c2**2))) * np.arctanh(1 / (c1*c2**2*R1*np.sqrt((c0**2 - c2**2) / ((c0**2 + vA**2)*(cT**2 - c2**2)*(vA**2 - c2**2)*(c1**2 - c2**2)))))
    elif c1 < c2:
        cutoff = 0.5 * np.sqrt(((c0**2 + vA**2) * (cT**2 - c1**2)) / ((c0**2 - c1**2) * (vA**2 - c1**2))) * np.arctanh(1 / (c2*c1**2*R2*np.sqrt((c0**2 - c1**2) / ((c0**2 + vA**2)*(cT**2 - c1**2)*(vA**2 - c1**2)*(c2**2 - c1**2)))))
    else:
        cutoff = 0
    return cutoff

def c1(R1):
    return c2*np.sqrt(R2/R1)

def trans(R1):
    c1 = c2*np.sqrt(R2/R1)
    trans = 0.5 * (c0**2*(R2*c2*np.sqrt(c1** - c0**2) + R1*c1*np.sqrt(c2**2-c0**2))) / ((vA**2-c0**2)*np.sqrt((c1**2-c0**2)*(c2**2-c0**2)))
    return trans
    
###############################################################################
# Set up the data

#number of iternations in R1 and K
NR1 = 500 #100
NK = 500 #400 for fast kink surf? #100

Kmax1 = 8. #5.
R1max1 = 4.

#initial and final K and R1 values
if mode == 'slow-kink-surf':
    Kmin = 0.01
    R1min = 0.001
    Kmax = 3.
    R1max = 4.
if mode == 'slow-saus-surf':
    Kmin = 0.01
    R1min = 0.5
    Kmax = 3.
    R1max = 3.5
if mode == 'fast-kink-surf':
    Kmin = 0.5
    R1min = 1.5
    Kmax = 15.
    R1max = 2.5
if mode == 'fast-saus-surf':
    Kmin = 0.1 # 0.6
    R1min = 1.5
    Kmax = Kmax1
    R1max = 2.5


Kvals = np.linspace(Kmin,Kmax,NK)
R1vals = np.linspace(R1min,R1max,NR1)
X, Y = np.meshgrid(Kvals,R1vals)

#Initialise R1 and W
R1 = R1min
def Wstart(mode):
    if mode == 'slow-kink-surf':
        return 0.1
    elif mode == 'slow-saus-surf':
        return cT - 0.004
    elif mode == 'fast-kink-surf':
        return c0 + 0.01
    elif mode == 'fast-saus-surf':
        return c1(R1) - 0.01


Wvals = np.zeros((NR1,NK))
Wvals[:] = np.NaN
data_set = np.zeros((NR1,NK))
data_set[:] = np.NaN

#for each K and for each R1 find the solution of the dispersion relation and put it in Wvals array.
W = Wstart(mode) #0.1
for i in range(0,NR1):
    if i != 0:
        W = Wstart(mode)#Wvals[i-1,0] #start at the solution valu eof the previous iteration
    R1 = R1vals[i]
    if mode == 'fast-saus-surf':
        co_int = int(np.ceil(cutoff(R1) * NK / (Kmax - Kmin)))
    elif mode == 'fast-kink-surf':
        co_int = int(np.ceil(trans(R1) * NK / (Kmax - Kmin)))
    else:
        co_int = 0
    for j in range(co_int,NK):
        K = Kvals[j]
        W = newton(partial(disp_rel_asym, K=K, R1=R1),W,tol=1e-5,maxiter=50)
        Wvals[i,j] = W
        print('Found solution number (' + str(i) + ',' + str(j) + ')')
        if plot_variable == 'amp-ratio':
            data_set[i,j] = amp_ratio(W, Kvals[j], R1vals[i], mode)
        elif plot_variable == 'min-pert-shift':
            data_set[i,j] = min_pert_shift(W, Kvals[j], R1vals[i], mode)
        elif plot_variable == 'W':
            data_set[i,j] = W
        else:
            print("'plot_variable' can only be 'amp-ratio' or 'min-pert-shift'")


for i in range(NR1):
    for j in range(NK):
        if abs(data_set[i,j]) == np.inf:
                data_set[i,j] = np.nan
        
font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 18}

matplotlib.rc('font', **font)

def levels(data,plot_variable,n):
    min_level = np.floor(np.nanmin(data))
    max_level = np.ceil(np.nanmax(data))
    largest = max(abs(max_level),abs(min_level))
    if plot_variable == 'amp-ratio':
        if 'kink' in mode:
            return np.linspace(2 - max_level, max_level, n)
        if 'saus' in mode:
            return np.linspace(min_level, -2 - min_level, n)
    elif plot_variable == 'min-pert-shift':
        return np.linspace(-largest, largest, n)
    elif plot_variable == 'W':
        return np.linspace(np.nanmin(data), np.nanmax(data), n)

#plt.imshow(data_set, cmap='coolwarm', origin='lower', interpolation='nearest')

fig = plt.figure()

contour = plt.contourf(Y, X, data_set, levels=levels(data_set,plot_variable,1000), 
                       cmap='RdBu')
plt.xlabel(r'$\rho_1/\rho_0$', fontsize=25)
plt.ylabel(r'$kx_0$', fontsize=25)
if mode == 'fast-kink-surf':
    plt.ylim([4.,Kmax])
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top') 

# get data you will need to create a "background patch" to your plot
width = R1max - R1min
height = Kmax - Kmin

ax = plt.gca()
rect = ax.patch
rect.set_facecolor((0.9,0.9,0.9))
#p = patches.Rectangle((R1min, Kmin), width, height, hatch='//', fill=None, zorder=-5)
#ax.add_patch(p)


def fmt(x, pos):
    a = '{:.1f}'.format(x)
    return a
cbar = plt.colorbar(format=matplotlib.ticker.FuncFormatter(fmt))
#cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)])
plt.tight_layout()
plt.show()

filename = mode + '_' + plot_variable
plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
            + filename)
