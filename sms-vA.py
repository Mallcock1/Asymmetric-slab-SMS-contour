# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:13:59 2017

@author: Matt
"""

import numpy as np
import scipy as sc
from scipy.optimize import newton
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# SBS
# Define the sound speeds and alfven speeds.
c2 = 1.2 #1.2
c0 = 1.
def cT(vA):
    return sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))
R2 = 2.
R1 = 1.5
c1 = c2*np.sqrt(R2/R1)

K = 1.
W = 0.6

mode_options = ['slow-kink-surf', 'slow-saus-surf', 'fast-kink-surf',
                'fast-saus-surf']

kink_mode_options = ['kink']
saus_mode_options = ['saus']

for mode in mode_options:
    if 'kink' in mode:
        kink_mode_options.append(mode)
    if 'saus' in mode:
        saus_mode_options.append(mode)

mode = mode_options[1]

method = 'amp-ratio'
method = 'min-pert-shift'

show_scatter = False
#show_scatter = True

###############################################################################

error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
error_string_subscript = "subscript argument must be 1 or 2"

def m0(W, vA):
    m0function = sc.sqrt((c0**2-W**2)*(vA**2-W**2)*((c0**2+vA**2)*(cT(vA)**2-W**2))**(-1))
    return m0function
    
def m1(W):
    m1function = sc.sqrt(1-W**2*c2**(-2)*R1*R2**(-1))
    return m1function

def m2(W):
    m2function = sc.sqrt(1-W**2*c2**(-2))
    return m2function

def disp_rel_sym(W, K, vA, mode, subscript):
    if subscript == 1:
        if mode in kink_mode_options:
            dispfunction = (vA**2 - W**2)*m1(W)*sc.tanh(m0(W,vA)*K) - R1*W**2*m0(W,vA)
        elif mode in saus_mode_options:
            dispfunction = (vA**2 - W**2)*m1(W) - R1*W**2*m0(W,vA)*sc.tanh(m0(W,vA)*K)
        else:
            print(error_string_kink_saus)
    elif subscript == 2:
        if mode in kink_mode_options:
            dispfunction = (vA**2 - W**2)*m2(W)*sc.tanh(m0(W,vA)*K) - R1*W**2*m0(W,vA)
        elif mode in saus_mode_options:
            dispfunction = (vA**2 - W**2)*m2(W) - R1*W**2*m0(W,vA)*sc.tanh(m0(W,vA)*K)
        else:
            print(error_string_kink_saus)
    else:
        print(error_string_subscript)
    return dispfunction
    
def disp_rel_asym(W, K, vA):
    return ((W**4 * m0(W,vA)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W) * m2(W) -
            0.5 * m0(W,vA) * W**2 * (vA**2 - W**2) * (R2 * m1(W) + R1 * m2(W)) *
            (sc.tanh(m0(W,vA) * K) + (sc.tanh(m0(W,vA) * K))**(-1))) /
            (vA**2 - W**2) * (c0**2 - W**2) * (cT(vA)**2 - W**2))
    
def amp_ratio(W, K, vA, mode):
    if mode in kink_mode_options:
        ampfunction = m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W)*disp_rel_sym(W,K,vA,'saus',2))
    elif mode in saus_mode_options:
        ampfunction = - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W)*disp_rel_sym(W,K,vA,'kink',2))
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def amp_ratio_func(W, K, vA, mode, RA):
    if mode in kink_mode_options:
        return m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W)*disp_rel_sym(W,K,vA,'saus',2)) - RA
    elif mode in saus_mode_options:
        return - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W)*disp_rel_sym(W,K,vA,'kink',2)) - RA
    else:
        print(error_string_kink_saus)
    
def min_pert_shift(W, K, vA, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA,'saus',1) / disp_rel_sym(W,K,vA,'kink',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA,'kink',1) / disp_rel_sym(W,K,vA,'saus',1))
    else:
        print(error_string_kink_saus)
    return np.real(shiftfunction)
    
def min_pert_shift_func(W, K, vA, mode, DM):
    return min_pert_shift(W, K, vA, mode) - DM
    
###############################################################################
# Set up the data
#number of iternations in RA or Delta min
NRA = 500
NDM = 500

RAmin = [0.98, 1.0005, 0.9, -1.0005, -0.97, -0.88]
DMmin = -0.9999999
RAmax = [0.3, 2., 0.6, -2., 1.5, -0.97345]
DMmax = 1.
vA_guess_list = [0.1, 1.3, 0.9, 1.3, 0.8, 0.697]

Wfix = 0.3
Kfix = 1.
modes = [0,1]

branches = [3,3]

fig = plt.figure()
for mode_ind in modes:
    for b in range(branches[mode_ind]):
        mode = mode_options[mode_ind]
        nb = sum(branches[:mode_ind]) + b
        
        RAvals = np.linspace(RAmin[nb], RAmax[nb], NRA)
        vAvals = np.zeros(NRA)
        vA_guess_init = vA_guess_list[nb]
        vA_guess = vA_guess_init
        for i in range(0,NRA):
            if i != 0:
                if vAvals[i-1] < 100:
                    vA_guess = vAvals[i-1]
                else:
                    va_guess = vA_guess_init
            vAvals[i] = newton(partial(amp_ratio_func, W, K, mode=mode, RA=RAvals[i]),vA_guess,tol=1e-5,maxiter=50)
            if vAvals[i] > 5:
                vAvals[i] = np.NaN
            print('Found solution number ' + str(i) + ' out of ' + str(NRA))
        plt.plot(RAvals, vAvals)

plt.plot([min(RAmin + RAmax), max(RAmin + RAmax)], [c1,c1],'k-.')
plt.plot([min(RAmin + RAmax), max(RAmin + RAmax)], [c0,c0],'k-.')
plt.plot([min(RAmin + RAmax), max(RAmin + RAmax)], [c2,c2],'k-.')
plt.plot([min(RAmin + RAmax), max(RAmin + RAmax)], [W,W],'k-.')

#DMvals = np.linspace(DMmin,DMmax,NDM)
#vAvals = np.zeros(NDM)
#vA_guess = 1.3
#for i in range(0,NDM):
#    if i != 0:
#        if vAvals[i-1] < 100:
#            vA_guess = vAvals[i-1]
#        else:
#            va_guess = 1.3
#    vAvals[i] = newton(partial(min_pert_shift_func, W, K, mode=mode, DM=DMvals[i]),vA_guess,tol=1e-5,maxiter=50)
#    if abs(vAvals[i]) > 5:
#        vAvals[i] = np.NaN
#    print('Found solution number ' + str(i) + ' out of ' + str(NDM))
#
#fig = plt.figure()
#plt.plot(DMvals, vAvals)


if show_scatter == True:
    NRA = 200
    NvA = 200
    
    RAmin = -2.
    RAmax = 2.
    vAmin = 0.3
    vAmax = 1.5
    
    RA_scatter_vals = np.linspace(RAmin, RAmax, NRA)
    vA_scatter_vals = np.linspace(vAmin, vAmax, NvA)
    
#    plt.figure()
    for mode in ['slow-kink-surf', 'slow-saus-surf']:
        vA = np.zeros(NRA * NvA)
        RA = np.zeros(NRA * NvA)
        vA[:] = np.NAN
        RA[:] = np.NAN        
        a=0
        for i in range(0,NRA):
            for j in range(0,NvA):
                if abs(amp_ratio_func(W, K, vA_scatter_vals[i], mode, RA_scatter_vals[j])) < 0.01:
                    vA[a] = vA_scatter_vals[i]
                    RA[a] = RA_scatter_vals[j]
                    a=a+1
        plt.scatter(RA, vA, marker='.')
#    plt.ylim([0.2, 1.6])
#    plt.xlim([-2., 2.])
    