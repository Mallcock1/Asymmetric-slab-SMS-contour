# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:13:59 2017

@author: Matt
"""

import numpy as np
import scipy as sc
from functools import partial
import matplotlib.pyplot as plt
import toolbox as tool

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

mode = mode_options[0]

show_RA = False
show_DM = False
show_scatter_RA = False
show_scatter_DM = False
show_scatter_DM_2 = False

show_RA = True
show_DM = True
#show_scatter_RA = True
#show_scatter_DM = True
#show_scatter_DM_2 = True

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
    
def amp_ratio_func(W, K, mode, vA, RA):
    if mode in kink_mode_options:
        return m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W)*disp_rel_sym(W,K,vA,'saus',2)) - RA
    elif mode in saus_mode_options:
        return - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W)*disp_rel_sym(W,K,vA,'kink',2)) - RA
    else:
        print(error_string_kink_saus)
    
def min_pert_shift(W, K, vA, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * np.arctanh(- disp_rel_sym(W,K,vA,'kink',1) / disp_rel_sym(W,K,vA,'saus',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W,vA)) * np.arctanh(- disp_rel_sym(W,K,vA,'saus',1) / disp_rel_sym(W,K,vA,'kink',1))
    else:
        print(error_string_kink_saus)
    return shiftfunction
    
def min_pert_shift_func(W, K, mode, vA, DM):
    return min_pert_shift(W, K, vA, mode) - DM
    
def min_pert_shift_2(W, K, vA, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * np.arctanh(disp_rel_sym(W,K,vA,'kink',2) / disp_rel_sym(W,K,vA,'saus',2))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W,vA)) * np.arctanh(disp_rel_sym(W,K,vA,'saus',2) / disp_rel_sym(W,K,vA,'kink',2))
    else:
        print(error_string_kink_saus)
    return shiftfunction
    
def min_pert_shift_func_2(W, K, mode, vA, DM):
    return min_pert_shift_2(W, K, vA, mode) - DM
    
###############################################################################

if show_RA == True:
    # Set up the data
    RAmin = [0.9703, 1.005, 0., -2., -0.961, -0.97348]
    RAmax = [0.9812, 2., 0.96, -1.01, 0., -0.9608]
    
    RA_guess = [0.971, 1.2, 0.88, -1.1, -0.8, -0.96]
    vA_guess = [0.55, 1.04, 0.9, 1.26, 1.04, 0.26]
    
    step = [0.0001, 0.0005, 0.001, 0.001, 0.001, 0.00002]
    
    modes = [0,1]
    
    branches = [3,3]
    
    styles = ['--'] * 3 + ['-'] * 3

    plt.figure()
    for mode_ind in modes:
        for b in range(branches[mode_ind]):
            mode = mode_options[mode_ind]
            nb = sum(branches[:mode_ind]) + b
            
            RA_values, root_array = tool.line_trace(partial(amp_ratio_func, W, K, mode), RA_guess[nb], 
                                                    vA_guess[nb], step[nb], RAmin[nb], RAmax[nb], (None))
            plt.plot(RA_values, root_array, linestyle=styles[nb], color='black')
            
            plt.plot(RA_values, disp_rel_asym(W, K, np.real(root_array)), color='red')
            
    ax = plt.gca()
    ax.fill_between((-2., 2.), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, color='lightgray')
    ax.set_ylabel(r'$v_\mathrm{A}$', fontsize = 20)
    ax.set_xlabel(r'$R_\mathrm{A}$', fontsize = 20)
    plt.ylim([0.,2.])
    plt.plot([0.,0.], [0., 2.], color='black', linestyle='-')
    ax.annotate('Body modes', xy=(1.2, 0.65), xycoords='data', annotation_clip=False, fontsize=12)
    ax.annotate('Kink', xy=(0.3, 1.8), xycoords='data', annotation_clip=False, fontsize=15)
    ax.annotate('Sausage', xy=(-0.8, 1.8), xycoords='data', annotation_clip=False, fontsize=15)


if show_DM == True:
    # Set up the data
    DMmin = [-0.975, -0.999]
    DMmax = [1., 1.]
    
    DM_guess = [-0.5, -0.5]
    vA_guess = [1.29, 0.9]
    
    step = [0.001, 0.001]
    
    modes = [0,1]
    
    branches = [1,1]
    
    styles = ['--', '-']
    
    plt.figure()
    for mode_ind in modes:
        for b in range(branches[mode_ind]):
            mode = mode_options[mode_ind]
            nb = sum(branches[:mode_ind]) + b
            
            DM_values, root_array = tool.line_trace(partial(min_pert_shift_func, W, K, mode), DM_guess[nb], 
                                                    vA_guess[nb], step[nb], DMmin[nb], DMmax[nb], (None))
            plt.plot(DM_values, root_array, linestyle=styles[nb], color='black')
            plt.plot(DM_values, disp_rel_asym(W, K, np.real(root_array)), color='red')
            
    ax = plt.gca()
    ax.fill_between((-1.2, -1.), (0.,0.), (2.,2.), color='lightgray')
    ax.fill_between((1., 1.2), (0.,0.), (2.,2.), color='lightgray')
    ax.fill_between((-2., 2.), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, color='lightgray')
    ax.set_ylabel(r'$v_\mathrm{A}$', fontsize = 20)
    ax.set_xlabel(r'$\Delta_\mathrm{min}$', fontsize = 20)
    plt.ylim([0.,2.])
    plt.xlim([-1.2,1.2])
    plt.plot([-1.,-1.], [0., 2.], color='black', linestyle='-')
    plt.plot([1.,1.], [0., 2.], color='black', linestyle='-')
    ax.annotate('Body modes', xy=(0.5, 0.65), xycoords='data', annotation_clip=False, fontsize=12)


if show_scatter_RA == True:
    NRA = 200
    NvA = 200
    
    RAmin = -2.
    RAmax = 2.
    vAmin = 1.5
    vAmax = 0.
    
    RA_scatter_vals = np.linspace(RAmin, RAmax, NRA)
    vA_scatter_vals = np.linspace(vAmin, vAmax, NvA)
    
    plt.figure()
    for mode in ['slow-kink-surf']:
        vA = np.zeros(NRA * NvA)
        RA = np.zeros(NRA * NvA)
        vA[:] = np.NAN
        RA[:] = np.NAN        
        a=0
        for i in range(0,NRA):
            for j in range(0,NvA):
                if abs(amp_ratio_func(W, K, mode, vA_scatter_vals[i], RA_scatter_vals[j])) < 0.01:
                    vA[a] = vA_scatter_vals[i]
                    RA[a] = RA_scatter_vals[j]
                    a=a+1
        plt.scatter(RA, vA, marker='.')
#    plt.ylim([0.2, 1.6])
#    plt.xlim([-2., 2.])


if show_scatter_DM == True:
    NDM = 200
    NvA = 200
    
    DMmin = -1.2
    DMmax = 1.2
    vAmin = 2.5
    vAmax = 0.
    
    DM_scatter_vals = np.linspace(DMmin, DMmax, NDM)
    vA_scatter_vals = np.linspace(vAmin, vAmax, NvA)
    
    modes = [0,1]
    
    plt.figure()
    for mode_ind in modes:
        mode = mode_options[mode_ind]
        vA = np.zeros(NDM * NvA)
        DM = np.zeros(NDM * NvA)
        vA[:] = np.NAN
        DM[:] = np.NAN     
        a=0
        for i in range(0,NDM):
            for j in range(0,NvA):
                if abs(min_pert_shift_func(W, K, mode, vA_scatter_vals[i], DM_scatter_vals[j])) < 0.01:
                    vA[a] = vA_scatter_vals[i]
                    DM[a] = DM_scatter_vals[j]
                    a=a+1
        if mode_ind == 0:
            plt.scatter(DM, vA, marker='.', color='black')
        if mode_ind == 1:
            plt.scatter(DM, vA, marker='.', color='red')
#    plt.ylim([0.2, 1.6])
#    plt.xlim([-2., 2.])


if show_scatter_DM_2 == True:
    NDM = 200
    NvA = 200
    
    DMmin = -1.2
    DMmax = 1.2
    vAmin = 2.5
    vAmax = 0.
    
    DM_scatter_vals = np.linspace(DMmin, DMmax, NDM)
    vA_scatter_vals = np.linspace(vAmin, vAmax, NvA)
    
    modes = [0,1]    
    
    plt.figure()
    for mode_ind in modes:
        mode = mode_options[mode_ind]
        vA = np.zeros(NDM * NvA)
        DM = np.zeros(NDM * NvA)
        vA[:] = np.NAN
        DM[:] = np.NAN     
        a=0
        for i in range(0,NDM):
            for j in range(0,NvA):
                if abs(min_pert_shift_func_2(W, K, mode, vA_scatter_vals[i], DM_scatter_vals[j])) < 0.01:
                    vA[a] = vA_scatter_vals[i]
                    DM[a] = DM_scatter_vals[j]
                    a=a+1
        if mode_ind == 0:
            plt.scatter(DM, vA, marker='.', color='black')
        if mode_ind == 1:
            plt.scatter(DM, vA, marker='.', color='red')
#    plt.ylim([0.2, 1.6])
#    plt.xlim([-2., 2.])