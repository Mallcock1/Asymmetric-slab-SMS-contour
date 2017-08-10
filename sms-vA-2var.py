# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:40:37 2017

@author: Matt
"""

import numpy as np
import scipy as sc
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import toolbox as tool
from scipy.optimize import fsolve

#def function(vA, R1):
#    return [amp_ratio_func(W, K, mode, vA, R1, RA),
#            amp_ratio_func_2(W, K, mode, vA, R1, RA)]
#fsolve(function, [vAguess, R1guess], xtol=1e-03)

# SBS
# Define the sound speeds and alfven speeds.
c2 = 1.2 #1.2
c0 = 1.
def cT(vA):
    return sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))
R2 = 2.
R1 = 1.5
#c1 = c2*np.sqrt(R2/R1)

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


show_RA = False
show_DM = False
show_scatter_RA = False
show_scatter_DM = False
#
show_RA = True
#show_DM = True
#show_scatter_RA = True
#show_scatter_DM = True

###############################################################################

def m0(W, vA):
    m0function = sc.sqrt((c0**2-W**2)*(vA**2-W**2)*((c0**2+vA**2)*(cT(vA)**2-W**2))**(-1))
    return m0function
    
def m1(W, R1):
    m1function = sc.sqrt(1-W**2*c2**(-2)*R1*R2**(-1))
    return m1function

def m2(W):
    m2function = sc.sqrt(1-W**2*c2**(-2))
    return m2function

def lamb0(W, vA):
    return -(vA**2-W**2)*1.j/(m0(W, vA)*W)

def lamb1(W, R1):
    return R1*W*1.j/m1(W, R1)
    
def lamb2(W):
    return R2*W*1.j/m2(W)

error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
error_string_subscript = "subscript argument must be 1 or 2"

def disp_rel_sym(W, K, vA, R1, mode, subscript):
    if subscript == 1:
        if mode in kink_mode_options:
            dispfunction = lamb0(W, vA) * sc.tanh(m0(W, vA) * K) + lamb1(W, R1)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W, vA) + lamb1(W, R1) * sc.tanh(m0(W, vA) * K)
        else:
            print(error_string_kink_saus)
    elif subscript == 2:
        if mode in kink_mode_options:
            dispfunction = lamb0(W, vA) * sc.tanh(m0(W, vA) * K) + lamb2(W)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W, vA) + lamb2(W) * sc.tanh(m0(W, vA) * K)
        else:
            print(error_string_kink_saus)
    else:
        print(error_string_subscript)
    return dispfunction
    
def disp_rel_asym(W, K, vA, R1):
    return ((W**4 * m0(W, vA)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W, R1) * m2(W) -
            0.5 * m0(W, vA) * W**2 * (vA**2 - W**2) * (R2 * m1(W, R1) + R1 * m2(W)) *
            (sc.tanh(m0(W, vA) * K) + (sc.tanh(m0(W, vA) * K))**(-1))) /
            (vA**2 - W**2) * (c0**2 - W**2) * (cT(vA)**2 - W**2))
            
def amp_ratio(W, K, vA, R1, mode):
    if mode in kink_mode_options:
        ampfunction = disp_rel_sym(W, K, vA, R1, 'saus', 1) / disp_rel_sym(W, K, vA, R1, 'saus', 2)
    elif mode in saus_mode_options:
        ampfunction = - disp_rel_sym(W, K, vA, R1, 'kink', 1) / disp_rel_sym(W, K, vA, R1, 'kink', 2)
    else:
        print(error_string_kink_saus)
    return ampfunction

def amp_ratio_func(W, K, mode, vA, R1, RA):
    return amp_ratio(W, K, vA, R1, mode) - RA
    
def amp_ratio_2(W, K, vA, R1, mode):
    if mode in kink_mode_options:
        ampfunction = - disp_rel_sym(W,K,vA, R1,'kink',1) / disp_rel_sym(W,K,vA, R1,'kink',2)
    elif mode in saus_mode_options:
        ampfunction = disp_rel_sym(W,K,vA, R1,'saus',1) / disp_rel_sym(W,K,vA, R1,'saus',2)
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def amp_ratio_func_2(W, K, mode, vA, R1, RA):
    return amp_ratio_2(W, K, vA, R1, mode) - RA    
    
def min_pert_shift(W, K, vA, R1, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA, R1,'saus',1) / disp_rel_sym(W,K,vA, R1,'kink',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA, R1,'kink',1) / disp_rel_sym(W,K,vA, R1,'saus',1))
    else:
        print(error_string_kink_saus)
    return shiftfunction
    
def min_pert_shift_func(W, K, mode, vA, R1, DM):
    return min_pert_shift(W, K, vA, R1, mode) - DM 
    
def min_pert_shift_2(W, K, vA, R1, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(disp_rel_sym(W,K, vA, R1,'saus',2) / disp_rel_sym(W,K, vA, R1,'kink',2))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W, vA)) * sc.arctanh(disp_rel_sym(W,K, vA, R1,'kink',2) / disp_rel_sym(W,K, vA, R1,'saus',2))
    else:
        print(error_string_kink_saus)
    return shiftfunction

def min_pert_shift_func_2(W, K, mode, vA, R1, DM):
    return min_pert_shift_2(W, K, vA, R1, mode) - DM 






























#error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
#error_string_subscript = "subscript argument must be 1 or 2"
#
#def m0(W, vA):
#    m0function = sc.sqrt((c0**2-W**2)*(vA**2-W**2)*((c0**2+vA**2)*(cT(vA)**2-W**2))**(-1))
#    return m0function
#    
#def m1(W):
#    m1function = sc.sqrt(1-W**2*c2**(-2)*R1*R2**(-1))
#    return m1function
#
#def m2(W):
#    m2function = sc.sqrt(1-W**2*c2**(-2))
#    return m2function
#
#def disp_rel_sym(W, K, vA, mode, subscript):
#    if subscript == 1:
#        if mode in kink_mode_options:
#            dispfunction = (vA**2 - W**2)*m1(W)*sc.tanh(m0(W,vA)*K) - R1*W**2*m0(W,vA)
#        elif mode in saus_mode_options:
#            dispfunction = (vA**2 - W**2)*m1(W) - R1*W**2*m0(W,vA)*sc.tanh(m0(W,vA)*K)
#        else:
#            print(error_string_kink_saus)
#    elif subscript == 2:
#        if mode in kink_mode_options:
#            dispfunction = (vA**2 - W**2)*m2(W)*sc.tanh(m0(W,vA)*K) - R1*W**2*m0(W,vA)
#        elif mode in saus_mode_options:
#            dispfunction = (vA**2 - W**2)*m2(W) - R1*W**2*m0(W,vA)*sc.tanh(m0(W,vA)*K)
#        else:
#            print(error_string_kink_saus)
#    else:
#        print(error_string_subscript)
#    return dispfunction
#    
#def disp_rel_asym(W, K, vA):
#    return ((W**4 * m0(W,vA)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W) * m2(W) -
#            0.5 * m0(W,vA) * W**2 * (vA**2 - W**2) * (R2 * m1(W) + R1 * m2(W)) *
#            (sc.tanh(m0(W,vA) * K) + (sc.tanh(m0(W,vA) * K))**(-1))) /
#            (vA**2 - W**2) * (c0**2 - W**2) * (cT(vA)**2 - W**2))
#
#def disp_rel_test(W, K, vA):
#    return disp_rel_sym(W, K, vA, 'saus',2) / disp_rel_sym(W, K, vA, 'saus',1) + disp_rel_sym(W, K, vA, 'kink',2) * disp_rel_sym(W, K, vA, 'kink',1)
#    
#def amp_ratio(W, K, vA, mode):
#    if mode in kink_mode_options:
#        ampfunction = m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W)*disp_rel_sym(W,K,vA,'saus',2))
#    elif mode in saus_mode_options:
#        ampfunction = - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W)*disp_rel_sym(W,K,vA,'kink',2))
#    else:
#        print(error_string_kink_saus)
#    return ampfunction
#    
#def amp_ratio_func(W, K, mode, vA, RA):
#    return amp_ratio(W, K, vA, mode) - RA
#
#def amp_ratio_2(W, K, vA, mode):
#    if mode in kink_mode_options:
#        ampfunction = - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W)*disp_rel_sym(W,K,vA,'kink',2))
#    elif mode in saus_mode_options:
#        ampfunction = m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W)*disp_rel_sym(W,K,vA,'saus',2))
#    else:
#        print(error_string_kink_saus)
#    return ampfunction
#    
#def amp_ratio_func_2(W, K, mode, vA, RA):
#    return amp_ratio_2(W, K, vA, mode) - RA
#    
#def min_pert_shift(W, K, vA, mode):
#    if mode in kink_mode_options:
#        shiftfunction = (1 / m0(W,vA)) * np.arctanh(- disp_rel_sym(W,K,vA,'kink',1) / disp_rel_sym(W,K,vA,'saus',1))
#    elif mode in saus_mode_options:
#        # recall that arccoth(x) = arctanh(1/x)
#        shiftfunction = (1 / m0(W,vA)) * np.arctanh(- disp_rel_sym(W,K,vA,'saus',1) / disp_rel_sym(W,K,vA,'kink',1))
#    else:
#        print(error_string_kink_saus)
#    return shiftfunction
#    
#def min_pert_shift_func(W, K, mode, vA, DM):
#    return min_pert_shift(W, K, vA, mode) - DM
#    
#def min_pert_shift_2(W, K, vA, mode):
#    if mode in kink_mode_options:
#        shiftfunction = (1 / m0(W,vA)) * np.arctanh(disp_rel_sym(W,K,vA,'kink',2) / disp_rel_sym(W,K,vA,'saus',2))
#    elif mode in saus_mode_options:
#        # recall that arccoth(x) = arctanh(1/x)
#        shiftfunction = (1 / m0(W,vA)) * np.arctanh(disp_rel_sym(W,K,vA,'saus',2) / disp_rel_sym(W,K,vA,'kink',2))
#    else:
#        print(error_string_kink_saus)
#    return shiftfunction
#    
#def min_pert_shift_func_2(W, K, mode, vA, DM):
#    return min_pert_shift_2(W, K, vA, mode) - DM
    
###############################################################################

font = {'size' : 15}
matplotlib.rc('font', **font)

if show_RA == True:
    # Set up the data
#    RAmin = [1.01, 0.0, 0.77, -2., -0.7944, -0.7205]
#    RAmax = [2.0, 0.69, 0.847, -1.1, -0.7, 0.0]
    
    R1_guess = [1.5, 1.5]
    vA_guess = [1.23, 1.23]
    
    RAmin = [2., -2.]
    RAmax = [-2., 2.]
    NRA = 500
    
#    step = [0.0005, 0.0005, 0.001, 0.001, 0.0001, 0.0001]
    
    modes = [0,1]#[0,1]
    
    branches = [1,1]#[3,3]
    
    styles = ['--'] + ['-']

    
    for mode_ind in modes:
        plt.figure()
        for b in range(branches[mode_ind]):
            mode = mode_options[mode_ind]
            nb = sum(branches[:mode_ind]) + b
            
            RA_vals = np.linspace(RAmin[nb], RAmax[nb], NRA)
            vA_sols = []            
            R1_sols = [] 
            RA_sols = []
            
            for i, RA in enumerate(RA_vals):
                def function(vA_R1):
                    return [amp_ratio_func(W, K, mode, vA_R1[0], vA_R1[1], RA),
                            amp_ratio_func_2(W, K, mode, vA_R1[0], vA_R1[1], RA)]
                if i != 0:
                    if abs(vA_sol) > 5.:
                        vA_guess[nb], R1_guess[nb] = 0.2, 0.2
                    else:
                        vA_guess[nb], R1_guess[nb] = vA_sol, R1_sol
                vA_sol, R1_sol = fsolve(function, [vA_guess[nb], R1_guess[nb]], xtol=1e-05)
                vA_sols.append(vA_sol)
                R1_sols.append(R1_sol)
                RA_sols.append(RA)
                
            
            plt.plot(RA_sols, vA_sols, color='black', linestyle=styles[nb])
            plt.plot(RA_sols, R1_sols, color='blue', linestyle=styles[nb])
            
            plt.plot(RA_sols, disp_rel_asym(W, K, np.array(vA_sols), np.array(R1_sols)), color='red', linestyle=styles[nb])
            
        ax = plt.gca()
        ax.fill_between((-2., 2.), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, color='lightgray')
        ax.set_ylabel(r'$v_\mathrm{A}$', fontsize = 20)
        ax.set_xlabel(r'$R_\mathrm{A}$', fontsize = 20)
        plt.ylim([-2.,5.])
        plt.axvline(color='black')
    #    ax.annotate('Body modes', xy=(1.2, 0.65), xycoords='data', annotation_clip=False, fontsize=12)
    #    ax.annotate('Quasi-kink', xy=(0.15, 1.8), xycoords='data', annotation_clip=False, fontsize=15)
    #    ax.annotate('Quasi-sausage', xy=(-1., 1.8), xycoords='data', annotation_clip=False, fontsize=15)
        plt.gcf().subplots_adjust(bottom=0.15)
    
#    filename = 'RA_vA_approx'
#    plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
#                + filename)

if show_DM == True:
    # Set up the data
    DMmin = [-0.999, -0.99]
    DMmax = [1., 1.]
    
    DM_guess = [-0.5, -0.5]
    vA_guess = [0.89, 1.29]
    
    step = [0.001, 0.0001]
    
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
#            plt.plot(DM_values, disp_rel_asym(W, K, np.real(root_array)), color='red')
            
    ax = plt.gca()
    ax.fill_between((-1.2, -1.), (0.,0.), (2.,2.), color='lightgray')
    ax.fill_between((1., 1.2), (0.,0.), (2.,2.), color='lightgray')
    ax.fill_between((-2., 2.), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, color='lightgray')
    ax.set_ylabel(r'$v_\mathrm{A}$', fontsize = 20)
    ax.set_xlabel(r'$\Delta_\mathrm{min} / x_0$', fontsize = 20)
    plt.ylim([0.,2.])
    plt.xlim([-1.2,1.2])
    plt.plot([-1.,-1.], [0., 2.], color='black', linestyle='-')
    plt.plot([1.,1.], [0., 2.], color='black', linestyle='-')
#    ax.annotate('Body modes', xy=(0.5, 0.65), xycoords='data', annotation_clip=False, fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.15)
#    
#    filename = 'DM_vA_approx'
#    plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
#                + filename)


if show_scatter_RA == True:
    NR1 = 500
    NRA = 10
    NvA = 50
    
    R1min = 0.1
    R1max = 10.
    RAmin = -2.
    RAmax = 2.
    vAmin = 0.
    vAmax = 2.
    
    R1_scatter_vals = np.linspace(R1min, R1max, NR1)
    RA_scatter_vals = np.linspace(RAmin, RAmax, NRA)
    vA_scatter_vals = np.linspace(vAmin, vAmax, NvA)
    
    plt.figure()
    for mode in ['slow-kink-surf']:
        vA = np.zeros(NRA * NvA)
        RA = np.zeros(NRA * NvA)
        vA[:] = np.NAN
        RA[:] = np.NAN        
        a=0
        for k in range(0,NR1):
            R1 = R1_scatter_vals[k]
            for i in range(0,NRA):
                for j in range(0,NvA):
                    print('vA = ' + str(vA_scatter_vals[j]))
                    if abs(amp_ratio_func(W, K, mode, vA_scatter_vals[j], R1, RA_scatter_vals[i])) < 0.1 and abs(amp_ratio_func_2(W, K, mode, vA_scatter_vals[j], R1, RA_scatter_vals[i])) < 0.1:
                        vA[a] = vA_scatter_vals[j]
                        RA[a] = RA_scatter_vals[i]
                        a=a+1
        plt.scatter(RA, vA, marker='.')
    plt.ylim([vAmin, vAmax])
    plt.xlim([RAmin, RAmax])


if show_scatter_DM == True:
    NDM = 200
    NvA = 200
    
    DMmin = -1.2
    DMmax = 1.2
    vAmin = 0.
    vAmax = 2.5
    
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