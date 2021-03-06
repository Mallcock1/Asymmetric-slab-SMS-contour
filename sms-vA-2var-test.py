# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:20:27 2017

@author: Matt
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:40:37 2017

@author: Matt
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import fsolve

# SBS
# Define the prescribed parameters.
c2 = 1.2
c0 = 1.
def cT(vA):
    return sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))

R2 = 2.
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

show_RA = True
#show_DM = True
#show_scatter_RA = True
#show_scatter_DM = True

###############################################################################

#def c1(vA, R1):
#    return np.sqrt(1/R1 * (c0**2 + 5./6 * vA**2))

def c2(vA, R2):
    return np.sqrt(1/R2 * (c0**2 + 5./6 * vA**2))

def m0(W, vA):
    m0function = sc.sqrt((c0**2-W**2)*(vA**2-W**2)*((c0**2+vA**2)*(cT(vA)**2-W**2))**(-1))
    return m0function
    
def m1(W, vA, R1, R2):
    m1function = sc.sqrt(1-W**2*c2(vA, R2)**(-2)*R1*R2**(-1))
    return m1function

def m2(W, vA, R2):
    m2function = sc.sqrt(1-W**2*c2(vA, R2)**(-2))
    return m2function

def lamb0(W, vA):
    return -(vA**2-W**2)*1.j/(m0(W, vA)*W)

def lamb1(W, vA, R1, R2):
    return R1*W*1.j/m1(W, vA, R1, R2)
    
def lamb2(W, vA, R2):
    return R2*W*1.j/m2(W, vA, R2)

error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
error_string_subscript = "subscript argument must be 1 or 2"

def disp_rel_sym(W, K, vA, R1, R2, mode, subscript):
    if subscript == 1:
        if mode in kink_mode_options:
            dispfunction = lamb0(W, vA) * sc.tanh(m0(W, vA) * K) + lamb1(W, vA, R1, R2)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W, vA) + lamb1(W, vA, R1, R2) * sc.tanh(m0(W, vA) * K)
        else:
            print(error_string_kink_saus)
    elif subscript == 2:
        if mode in kink_mode_options:
            dispfunction = lamb0(W, vA) * sc.tanh(m0(W, vA) * K) + lamb2(W, vA, R2)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W, vA) + lamb2(W, vA, R2) * sc.tanh(m0(W, vA) * K)
        else:
            print(error_string_kink_saus)
    else:
        print(error_string_subscript)
    return dispfunction
    
def disp_rel_asym(W, K, vA, R1, R2):
    return ((W**4 * m0(W, vA)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W, vA, R1, R2) * m2(W, vA, R2) -
            0.5 * m0(W, vA) * W**2 * (vA**2 - W**2) * (R2 * m1(W, vA, R1, R2) + R1 * m2(W, vA, R2)) *
            (sc.tanh(m0(W, vA) * K) + (sc.tanh(m0(W, vA) * K))**(-1))) /
            (vA**2 - W**2) * (c0**2 - W**2) * (cT(vA)**2 - W**2))
            
def amp_ratio(W, K, vA, R1, R2, mode):
    if mode in kink_mode_options:
        ampfunction = disp_rel_sym(W, K, vA, R1, R2, 'saus', 1) / disp_rel_sym(W, K, vA, R1, R2, 'saus', 2)
    elif mode in saus_mode_options:
        ampfunction = - disp_rel_sym(W, K, vA, R1, R2, 'kink', 1) / disp_rel_sym(W, K, vA, R1, R2, 'kink', 2)
    else:
        print(error_string_kink_saus)
    return ampfunction

W_star = W * c0 / (np.sqrt(c0**2 - W**2))

def amp_ratio_func(W, K, mode, vA, R1, R2, RA):
    return amp_ratio(W, K, vA, R1, R2, mode) - RA
    
def amp_ratio_2(W, K, vA, R1, R2, mode):
    if mode in kink_mode_options:
        ampfunction = - disp_rel_sym(W,K,vA, R1, R2,'kink',1) / disp_rel_sym(W,K,vA, R1, R2,'kink',2)
    elif mode in saus_mode_options:
        ampfunction = disp_rel_sym(W,K,vA, R1, R2,'saus',1) / disp_rel_sym(W,K,vA, R1, R2,'saus',2)
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def amp_ratio_func_2(W, K, mode, vA, R1, R2, RA):
    return amp_ratio_2(W, K, vA, R1, R2, mode) - RA
    
def min_pert_shift(W, K, vA, R1, R2, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA, R1, R2,'saus',1) / disp_rel_sym(W,K,vA, R1, R2,'kink',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(- disp_rel_sym(W,K,vA, R1, R2,'kink',1) / disp_rel_sym(W,K,vA, R1, R2,'saus',1))
    else:
        print(error_string_kink_saus)
    return shiftfunction
    
def min_pert_shift_func(W, K, mode, vA, R1, R2, DM):
    return min_pert_shift(W, K, vA, R1, R2, mode) - DM 
    
def min_pert_shift_2(W, K, vA, R1, R2, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W,vA)) * sc.arctanh(disp_rel_sym(W,K, vA, R1, R2,'saus',2) / disp_rel_sym(W,K, vA, R1, R2,'kink',2))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W, vA)) * sc.arctanh(disp_rel_sym(W,K, vA, R1, R2,'kink',2) / disp_rel_sym(W,K, vA, R1, R2,'saus',2))
    else:
        print(error_string_kink_saus)
    return shiftfunction

def min_pert_shift_func_2(W, K, mode, vA, R1, R2, DM):
    return min_pert_shift_2(W, K, vA, R1, R2, mode) - DM 

###############################################################################

font = {'size' : 15}
matplotlib.rc('font', **font)

if show_RA == True:

    R1_guess = [1.4, 2.4]
    vA_guess = [1.23, 1.23]
    
    RAmin = [2., -2.]
    RAmax = [-2., 2.]
    NRA = 200
    
    modes = [0,1]
    
    branches = [1,1]
    
    styles = ['--'] * branches[0] + ['-'] * branches[1]
    
    param_vals = np.linspace(1., 0.1, 11) # 1.,3., 11
    
    for r,p in enumerate(param_vals):
        K = p
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        if r==0:
            R1_guess0 = R1_guess[0]
            R1_guess1 = R1_guess[1]
            vA_guess0 = vA_guess[0]
            vA_guess1 = vA_guess[1]
        
        ln1 = []
        ln2 = []
        for mode_ind in modes:
            
            for b in range(branches[mode_ind]):
                mode = mode_options[mode_ind]
                nb = sum(branches[:mode_ind]) + b
                
                RA_vals = np.linspace(RAmin[nb], RAmax[nb], NRA)
                vA_sols = []            
                R1_sols = [] 
                RA_sols = []
                
                for i, RA in enumerate(RA_vals):
                    def function(vA_R1):
                        return [amp_ratio_func(W, K, mode, vA_R1[0], vA_R1[1], R2, RA),
                                amp_ratio_func_2(W, K, mode, vA_R1[0], vA_R1[1], R2, RA)]
                    if i != 0:
                        if (mode_ind == 0 and RA < 0.) or (mode_ind == 1 and RA > 0.):
                            break
                        else:
                            vA_guess[nb], R1_guess[nb] = vA_sol, R1_sol
                    vA_sol, R1_sol = fsolve(function, [vA_guess[nb], R1_guess[nb]], xtol=1e-08)
                    vA_sols.append(vA_sol)
                    R1_sols.append(R1_sol)
                    RA_sols.append(RA)
                

                R1_guess[nb] = R1_sols[0]
                vA_guess[nb] = vA_sols[0]
                
                if mode_ind == 1:
                    ln1 = ln1 + ax1.plot(RA_sols, vA_sols, color='black',
                                         linestyle=styles[nb], label=r'$v_\mathrm{A}$')
                    ln2 = ln2 + ax2.plot(RA_sols, R1_sols, color='blue', 
                                         linestyle=styles[nb], label=r'$\rho_1 / \rho_0$')
                else:
                    ax1.plot(RA_sols, vA_sols, color='black', linestyle=styles[nb])
                    ax2.plot(RA_sols, R1_sols, color='blue', linestyle=styles[nb])
                plt.plot(RA_sols, disp_rel_asym(W, K, np.array(vA_sols), np.array(R1_sols), R2), 
                         color='red', linestyle=styles[nb])
            
            ax1.set_ylim([0.,3.])
            ax2.set_ylim([0.,3.])
            ax1.fill_between((-2., 2.), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, 
                             color='lightgray')
            ax1.set_ylabel(r'$v_\mathrm{A}/c_0$', fontsize = 20)
            ax1.set_xlabel(r'$R_\mathrm{A}$', fontsize = 20)
            ax2.set_ylabel(r'$\rho_1 / \rho_0$', fontsize = 20)
            
            lns = ln1+ln2
            labs = [l.get_label() for l in lns]
            plt.legend(lns, labs, loc=4, fancybox=True, framealpha=0.7)
            
            plt.axvline(color='black')
            plt.gcf().subplots_adjust(bottom=0.15)
        
        
            textstr = (r'$\rho_2/\rho_0=%.1f$' + '\n' + r'$\omega/k c_0=%.1f$' + '\n' + 
                       r'$kx_0=%.1f$')%(R2, W, K)

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            
            # place a text box in upper left in axes coords
            ax = plt.gca()
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=17,
                    verticalalignment='top', bbox=props)
    #        filename = 'RA_vA_approx_2var'
    #        plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
    #                    + filename)


if show_DM == True:
    
    R1_guess = [1.5, 0.1]
    vA_guess = [1.23, 1.23]
    
    DMmin = [0.001, -0.999]
    DMmax = [1., 1.2]
    NDM = 500
    
#    step = [0.0005, 0.0005, 0.001, 0.001, 0.0001, 0.0001]
    
    modes = [0,1]#[0,1]
    
    branches = [1,1]#[3,3]
    
    styles = ['--'] + ['-']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ln1 = []
    ln2 = []
    for mode_ind in modes:
        for b in range(branches[mode_ind]):
            mode = mode_options[mode_ind]
            nb = sum(branches[:mode_ind]) + b
            
            DM_vals = np.linspace(DMmin[nb], DMmax[nb], NDM)
            vA_sols = []            
            R1_sols = [] 
            DM_sols = []
            
            for i, DM in enumerate(DM_vals):
                def function(vA_R1):
                    return [min_pert_shift_func(W, K, mode, vA_R1[0], vA_R1[1], DM),
                            min_pert_shift_func_2(W, K, mode, vA_R1[0], vA_R1[1], DM)]
                if i != 0:
                    if abs(vA_sol) > 5. or (mode_ind == 0 and DM < 0.) or (mode_ind == 1 and DM > 0.):
                        break
                    else:
                        vA_guess[nb], R1_guess[nb] = vA_sol, R1_sol
                vA_sol, R1_sol = fsolve(function, [vA_guess[nb], R1_guess[nb]], xtol=1e-05)
                vA_sols.append(vA_sol)
                R1_sols.append(R1_sol)
                DM_sols.append(DM)
            
            if mode_ind == 1:
                ln1 = ln1 + ax1.plot(DM_sols, vA_sols, color='black', linestyle=styles[nb], label=r'$v_\mathrm{A}$')
                ln2 = ln2 + ax2.plot(DM_sols, R1_sols, color='blue', linestyle=styles[nb], label=r'$\rho_1 / \rho_0$')
            else:
                ax1.plot(DM_sols, vA_sols, color='black', linestyle=styles[nb])
                ax2.plot(DM_sols, R1_sols, color='blue', linestyle=styles[nb])
            plt.plot(DM_sols, disp_rel_asym(W, K, np.array(vA_sols), np.array(R1_sols)), color='red', linestyle=styles[nb])
            
        ax1.set_ylim([0., 3.])
        ax2.set_ylim([0., 3.])
        plt.xlim([-1.2,1.2])
        ax1.fill_between((-1.2, 1.2), (W, W), [W * c0 / (np.sqrt(c0**2 - W**2))] * 2, color='lightgray')
        ax1.set_ylabel(r'$v_\mathrm{A}/c_0$', fontsize = 20)
        ax1.set_xlabel(r'$\Delta_\mathrm{min}/x_0$', fontsize = 20)
        ax2.set_ylabel(r'$\rho_1 / \rho_0$', fontsize = 20)
        ax1.fill_between((-1.2, -1.), (0.,0.), (5.,5.), color='lightgray')
        ax1.fill_between((1., 1.2), (0.,0.), (5.,5.), color='lightgray')
        
        
        lns = ln1+ln2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=4, fancybox=True, framealpha=0.7)
        
        plt.axvline(color='black')
        plt.axvline(x=-1., color='black')
        plt.axvline(x=1., color='black')
    #    ax.annotate('Body modes', xy=(1.2, 0.65), xycoords='data', annotation_clip=False, fontsize=12)
    #    ax.annotate('Quasi-kink', xy=(0.15, 1.8), xycoords='data', annotation_clip=False, fontsize=15)
    #    ax.annotate('Quasi-sausage', xy=(-1., 1.8), xycoords='data', annotation_clip=False, fontsize=15)
        plt.gcf().subplots_adjust(bottom=0.15)
    
        filename = 'DM_vA_approx_2var'
        plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
                    + filename)


if show_scatter_RA == True:
    NR1 = 50
    NRA = 50
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
            print('k = ' + str(k) + ' out of ' + str(NR1))
            R1 = R1_scatter_vals[k]
            for i in range(0,NRA):
                for j in range(0,NvA):
                    if abs(amp_ratio_func(W, K, mode, vA_scatter_vals[j], R1, RA_scatter_vals[i])) < 0.1 and abs(amp_ratio_func_2(W, K, mode, vA_scatter_vals[j], R1, RA_scatter_vals[i])) < 0.1:
                        vA[a] = vA_scatter_vals[j]
                        RA[a] = RA_scatter_vals[i]
                        a=a+1
        plt.scatter(RA, vA, marker='.')
    plt.ylim([vAmin, vAmax])
    plt.xlim([RAmin, RAmax])


if show_scatter_DM == True:
    NR1 = 50    
    NDM = 50
    NvA = 50

    R1min = 0.1
    R1max = 10.
    DMmin = -1.2
    DMmax = 1.2
    vAmin = 0.
    vAmax = 2.5
    
    R1_scatter_vals = np.linspace(R1min, R1max, NR1)
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
        for k in range(0,NR1):
            print('R1 = ' + str(R1))
            R1 = R1_scatter_vals[k]
            for i in range(0,NDM):
                for j in range(0,NvA):
                    if abs(min_pert_shift_func(W, K, mode, vA_scatter_vals[j], R1, DM_scatter_vals[i])) < 0.1 and abs(min_pert_shift_func_2(W, K, mode, vA_scatter_vals[j], R1, DM_scatter_vals[i])) < 0.1:
                        vA[a] = vA_scatter_vals[j]
                        DM[a] = DM_scatter_vals[i]
                        a=a+1
            if mode_ind == 0:
                plt.scatter(DM, vA, marker='.', color='black')
            if mode_ind == 1:
                plt.scatter(DM, vA, marker='.', color='red')
#    plt.ylim([0.2, 1.6])
#    plt.xlim([-2., 2.])


