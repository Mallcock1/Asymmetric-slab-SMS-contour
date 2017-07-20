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
c2 = 1.2
c0 = 1.
def cT(vA):
    return sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))
R2 = 2.
R1 = 1.5
c1 = c2*np.sqrt(R2/R1)

K = 1.
W = 0.3

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

method = 'amp-ratio'
method = 'min-pert-shift'

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
    
def disp_rel_asym(W, K, vA):
    return ((W**4 * m0(W)**2 * R1 * R2 + (vA**2 - W**2)**2 * m1(W,R1) * m2(W) -
            0.5 * m0(W) * W**2 * (vA**2 - W**2) * (R2 * m1(W,R1) + R1 * m2(W)) *
            (sc.tanh(m0(W) * K) + (sc.tanh(m0(W) * K))**(-1))) /
            (vA**2 - W**2) * (c0**2 - W**2) * (cT(vA)**2 - W**2))
    
def amp_ratio(W, K, vA, mode):
    if mode in kink_mode_options:
        ampfunction = m2(W)*disp_rel_sym(W,K,vA,'saus',1) / (m1(W,R1)*disp_rel_sym(W,K,vA,'saus',2))
    elif mode in saus_mode_options:
        ampfunction = - m2(W)*disp_rel_sym(W,K,vA,'kink',1) / (m1(W,R1)*disp_rel_sym(W,K,vA,'kink',2))
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def min_pert_shift(W, K, vA, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W)) * sc.arctanh(- disp_rel_sym(W,K,vA,'saus',1) / disp_rel_sym(W,K,vA,'kink',1))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W)) * sc.arctanh(- disp_rel_sym(W,K,vA,'kink',1) / disp_rel_sym(W,K,vA,'saus',1))
    else:
        print(error_string_kink_saus)
    return shiftfunction
    
###############################################################################
# Set up the data

#number of iternations in RA or Delta min
NRA = 50
NDM = 50

RAmin = 0.5
DMmin = -0.5
RAmax = 1.5
DMmax = 0.5


RAvals = np.linspace(RAmin,RAmax,NRA)
vAvals = np.zeros(NRA)
vA_guess = 1.3
for i in range(0,NRA):
    vAvals[i] = newton(partial(disp_rel_asym, W=W, K=K),vA_guess,tol=1e-5,maxiter=50)

plt.plot(vAvals)


        
#font = {'family' : 'normal',
#        'weight' : 'light',
#        'size'   : 18}
#
#matplotlib.rc('font', **font)
#
#def clim(data,plot_variable):
#    min_level = np.floor(np.nanmin(data))
#    max_level = np.ceil(np.nanmax(data))
#    largest = max(abs(max_level),abs(min_level))
#    if plot_variable == 'amp-ratio':
#        if 'kink' in mode:
#            return [2 - max_level, max_level]
#        if 'saus' in mode:
#            return [min_level, -2 - min_level]
#    elif plot_variable == 'min-pert-shift':
#        return [-largest, largest]
#    elif plot_variable == 'W':
#        return [np.nanmin(data), np.nanmax(data)]
#
#if plot_variable == 'amp-ratio' or plot_variable == 'W':
#    cmap = 'RdBu'
#else:
#    cmap = 'RdBu'
#
#fig = plt.figure()
#aspect = 'auto'#(R1vals[-1] - R1vals[0])/(Kvals[-1] - Kvals[0])# * 0.5
#im = plt.imshow(data_set.transpose(), cmap=cmap, origin='lower', clim=clim(data_set,plot_variable), 
#                aspect=aspect, extent=[R1vals[0],R1vals[-1],Kvals[0],Kvals[-1]])
#ax = plt.gca()
#
#
#plt.xlabel(r'$\rho_1/\rho_0$', fontsize=25)
#plt.ylabel(r'$kx_0$', fontsize=25)
#if mode == 'fast-kink-surf':
#    plt.ylim([4.,Kmax])
###ax.xaxis.tick_top()
###ax.xaxis.set_label_position('top') 
##
## get data you will need to create a "background patch" to your plot
#width = R1max - R1min
#height = Kmax - Kmin
#
## set background colour - NaNs have this colour
#rect = ax.patch
#rect.set_facecolor((0.9,0.9,0.9))
#
#def pad(data):
#    ppad = 38
#    mpad = 40
#    size_bool = np.where(data < 0)[0].size == 0
#    if size_bool:
#        p = ppad
#    else:
#        p = mpad
#    return p
#
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="7.5%", pad=0.2, aspect=13.33)
#
#def fmt(x, pos):
#    a = '{:.1f}'.format(x)
#    return a
##cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(cbar_min, cbar_max+cbar_step, cbar_step)])
#cb = plt.colorbar(im, cax=cax, 
#                  format=matplotlib.ticker.FuncFormatter(fmt))
#        
#ticklabs = cb.ax.get_yticklabels()
#cb.ax.set_yticklabels(ticklabs,ha='right')
#cb.ax.yaxis.set_tick_params(pad=pad(data_set))  # your number may vary
#
#plt.gcf().subplots_adjust(bottom=0.15)
##plt.tight_layout()
#plt.show()
#
#filename = mode + '_' + plot_variable
#plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
#            + filename)
