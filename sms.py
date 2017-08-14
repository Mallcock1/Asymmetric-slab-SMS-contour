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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shiftedcolormap as scm

# SBS
# Define the sound speeds and alfven speeds.
c2 = 1.2
c0 = 1.
vA = 1.3
cT = sc.sqrt(c0**2*vA**2*(c0**2+vA**2)**(-1))
R2 = 2.

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

plot_variable = 'amp-ratio'
#plot_variable = 'amp-ratio-2'
#plot_variable = 'min-pert-shift'
#plot_variable = 'min-pert-shift-2'
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

def lamb0(W):
    return -(vA**2-W**2)*1.j/(m0(W)*W)

def lamb1(W, R1):
    return R1*W*1.j/m1(W, R1)
    
def lamb2(W):
    return R2*W*1.j/m2(W)

error_string_kink_saus = "mode must be 'kink' or 'saus', duh!"
error_string_subscript = "subscript argument must be 1 or 2"

def disp_rel_sym(W, K, R1, mode, subscript):
    if subscript == 1:
        if mode in kink_mode_options:
            dispfunction = lamb0(W) * sc.tanh(m0(W) * K) + lamb1(W,R1)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W) + lamb1(W,R1) * sc.tanh(m0(W) * K)
        else:
            print(error_string_kink_saus)
    elif subscript == 2:
        if mode in kink_mode_options:
            dispfunction = lamb0(W) * sc.tanh(m0(W) * K) + lamb2(W)
        elif mode in saus_mode_options:
            dispfunction = lamb0(W) + lamb2(W) * sc.tanh(m0(W) * K)
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
        ampfunction = disp_rel_sym(W,K,R1,'saus',1) / disp_rel_sym(W,K,R1,'saus',2)
    elif mode in saus_mode_options:
        ampfunction = - disp_rel_sym(W,K,R1,'kink',1) / disp_rel_sym(W,K,R1,'kink',2)
    else:
        print(error_string_kink_saus)
    return ampfunction
    
def amp_ratio_2(W, K, R1, mode):
    if mode in kink_mode_options:
        ampfunction = - disp_rel_sym(W,K,R1,'kink',1) / disp_rel_sym(W,K,R1,'kink',2)
    elif mode in saus_mode_options:
        ampfunction = disp_rel_sym(W,K,R1,'saus',1) / disp_rel_sym(W,K,R1,'saus',2)
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
    
def min_pert_shift_2(W, K, R1, mode):
    if mode in kink_mode_options:
        shiftfunction = (1 / m0(W)) * sc.arctanh(disp_rel_sym(W,K,R1,'saus',2) / disp_rel_sym(W,K,R1,'kink',2))
    elif mode in saus_mode_options:
        # recall that arccoth(x) = arctanh(1/x)
        shiftfunction = (1 / m0(W)) * sc.arctanh(disp_rel_sym(W,K,R1,'kink',2) / disp_rel_sym(W,K,R1,'saus',2))
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
NR1 = 70 #100
NK = 70 #400 for fast kink surf? #100

Kmax1 = 8. #5.
R1max1 = 4.

#initial and final K and R1 values
if mode == 'slow-kink-surf':
    Kmin = 0.01
    R1min = 0.5 #0.001
    Kmax = 3.#3.
    R1max = 3.5 #4.
if mode == 'slow-saus-surf':
    Kmin = 0.01
    R1min = 0.5
    Kmax = 3.#3.
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
        elif plot_variable == 'amp-ratio-2':
            data_set[i,j] = amp_ratio_2(W, Kvals[j], R1vals[i], mode)
        elif plot_variable == 'min-pert-shift':
            data_set[i,j] = min_pert_shift(W, Kvals[j], R1vals[i], mode)
        elif plot_variable == 'min-pert-shift-2':
            data_set[i,j] = min_pert_shift_2(W, Kvals[j], R1vals[i], mode)
        elif plot_variable == 'W':
            data_set[i,j] = W
        else:
            print("'plot_variable' can only be 'amp-ratio' or 'min-pert-shift'")


for i in range(NR1):
    for j in range(NK):
        if abs(data_set[i,j]) == np.inf:
                data_set[i,j] = np.nan
        
font = {'size' : 15}
matplotlib.rc('font', **font)

# set limits of colormap range
def clim(data, plot_variable):
    min_level = round(np.nanmin(data),1)
    max_level = round(np.nanmax(data),1)
    if np.nanmin(data) < min_level:
        min_level= min_level - 0.1
    if np.nanmax(data) < max_level:
        max_level= max_level + 0.1
    largest = max(abs(max_level),abs(min_level))
    if 'amp-ratio' in plot_variable:    
        return None
    elif 'min-pert-shift' in plot_variable:
        return [-largest, largest]
    elif plot_variable == 'W':
        return [np.nanmin(data), np.nanmax(data)]

# set colormap and other kwargs for imshow
if 'amp-ratio' in plot_variable:
    mi = np.nanmin(data_set)
    if 'kink' in mode:
        ma = np.nanmax(data_set)
    if 'saus' in mode:
        if mode == 'slow-saus-surf':
            ma = -0.000001
        else:
            ma = np.nanmax(data_set)

    if 'saus' in mode:
        midpoint = - np.log10(-mi) / np.log10(ma/mi)
    elif 'kink' in mode:
        midpoint = - np.log10(mi) / np.log10(ma/mi)

    if 'saus' in mode and 'amp-ratio' in plot_variable:
        orig_cmap = matplotlib.cm.RdBu
    else:
        orig_cmap = matplotlib.cm.RdBu_r
    cmap = scm.shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')
elif 'min-pert-shift' in plot_variable:
    cmap = 'PuOr_r'
    

if 'amp-ratio' in plot_variable:
    if 'saus' in mode:
        norm = matplotlib.colors.SymLogNorm(linthresh=0.000000000001, linscale=0.000000000001, vmax=ma)
    if 'kink' in mode:
        norm = matplotlib.colors.LogNorm()
else:
    norm = None
        


fig = plt.figure()
aspect = 'auto'

# plot heat map
im = plt.imshow(data_set.transpose(), cmap=cmap, origin='lower', norm=norm, 
                clim=clim(data_set,plot_variable), aspect=aspect, 
                extent=[R1vals[0],R1vals[-1],Kvals[0],Kvals[-1]])


ax = plt.gca()

plt.xlabel(r'$\rho_1/\rho_0$', fontsize=25)
plt.ylabel(r'$kx_0$', fontsize=25)
if mode == 'fast-kink-surf':
    plt.ylim([4.,Kmax])
    
# get data you will need to create a "background patch" to your plot
width = R1max - R1min
height = Kmax - Kmin

# set background colour - NaNs have this colour
rect = ax.patch
rect.set_facecolor((0.9,0.9,0.9))

# set size/shape of colorbar axis
def pad(data):
    ppad = 38
    mpad = 40
    size_bool = np.where(data < 0)[0].size == 0
    if size_bool:
        p = ppad
    else:
        p = mpad
    return p
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="7.5%", pad=0.2, aspect=13.33)

# format colorbar ticks to 1 decimal place
def fmt(x, pos):
    a = '{:.1f}'.format(x)
    return a
cb = plt.colorbar(im, cax=cax, format=matplotlib.ticker.FuncFormatter(fmt))

# Colorbar label
if 'amp-ratio' in plot_variable:
    cb_label = r'$R_\mathrm{A}$'
elif 'min-pert-shift' in plot_variable:
    cb_label = r'$\Delta_\mathrm{min}$'
cb.set_label(cb_label, fontsize=25)


# colorbar tick labels
if 'amp-ratio' in plot_variable:
    if 'kink' in mode:
        cb_ticks = [0.01,0.1,1.,10.,100]
        cb_ticklabels = [r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$']
    elif 'saus' in mode:
        cb_ticks = [-100, -10, -1, -0.1, -0.01, -0.001, -0.0001, -0.00001, -0.000001]
        cb_ticklabels = [r'$-10^{2}$',r'$-10^{1}$',r'$-10^{0}$',r'$-10^{-1}$',
                           r'$-10^{-2}$',r'$-10^{-3}$',r'$-10^{-4}$',r'$-10^{-5}$',
                           r'$-10^{-6}$']
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels(cb_ticklabels)
else:
    cb_ticks = []
    cb_ticklabels = []
    for t in cb.ax.get_yticklabels():
        cb_ticks.append(float(t.get_text()))
        cb_ticklabels.append(t.get_text())

# overlay individual contours
contours = ax.contour(data_set.transpose(), levels=cb_ticks, colors='black', 
                       extent=[R1vals[0],R1vals[-1],Kvals[0],Kvals[-1]])

#Make negative contours solid too.
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#set contour labels
fmt = {}
for l, t in zip(cb_ticks, cb_ticklabels):
    fmt[l] = t
ax.clabel(contours, fmt=fmt, fontsize=14)


# invert y-axis for sausage modes (since they are negative values)
if 'saus' in mode and 'amp-ratio' in plot_variable:
    cb.ax.invert_yaxis()
    print('axis inverted')



#plt.gcf().subplots_adjust(bottom=0.15, right=3.)
plt.tight_layout()
plt.show()

filename = mode + '_' + plot_variable
plt.savefig('D:\\my_work\\projects\\Asymmetric_slab\\Python\\sms\\sms-plots\\' 
            + filename)
