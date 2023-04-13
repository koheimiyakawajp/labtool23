#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import sys
from astropy.visualization import simple_norm

pi  = np.pi
h   = 6.62607015e-34    #Js
c   = 2.99792485e+8     #m/s
k   = 1.380649e-23      #J/K

def flux_furnace(D, R, I):
    # D : distandce to the detector [m]
    # R : radius of the furnace [m]
    #
    val     = 2*pi*I*D*(1/D - 1/(R**2 + D**2)**0.5)
    return val
    #J/s/m^3

def flux_furnace_test(D,R,I):
    r   = np.linspace(0,R,1000)
    dr  = r[1] - r[0]

    v1  = (r+dr)**2 - r**2
    v2  = D/((D**2 + r**2)**(3./2.))

    sm  = np.sum(v1*v2)
    return I*pi*sm

def planck_func(T, lamb):
    # T : temperature

    T   = T
    c1  = 2.*h*(c**2)/(lamb**5)
    c2  = 1./(np.exp(h*c/(k*T*lamb)) - 1.)

    return c1*c2
    #J/s/m^3/sr


def obs_nphoton(Teff, filter, R=0.015, D=0.1):

    #qe          = 0.80 #
    qe          = 1.00 #
    size_pix    = 15e-6 #m

    lamb    = filter[0]

    I   = planck_func(Teff, lamb)
    irr = flux_furnace(D,R, I)
    obs = irr*qe*filter[1]

    dw  = np.median(lamb[1:] - lamb[:-1])
    e_photo = h*c/lamb
    dp  = obs[1:]*dw/e_photo[1:]
    n_s_m2  = np.sum(dp) #s/m^2
    n_s     = n_s_m2*(size_pix**2)

    return n_s

from scipy.interpolate import interp1d
def filter_set(filter_array, x):
    wav = filter_array[0]*1e-9
    trs = filter_array[1]*1e-2
    #f   = interp1d(wav, trs, kind="cubic")
    f   = interp1d(wav, trs, kind="linear")
    x_narrow    = x[((wav[-1] < x)&(x < wav[0]))]
    return x_narrow, f(x_narrow)

eV_J    = 1.60218e-19
if __name__=='__main__':

    filefilter  = sys.argv[1]
    filterdata  = np.loadtxt(filefilter, dtype='f8').T

    lamb        = np.linspace(0.8e-6, 1.8e-6, int(1e4))
    filter      = filter_set(filterdata, lamb)

    dist        = 1. #m
    rad         = 0.001 #m (radius of pinhole)

    print(obs_nphoton(1400+273, filter, R=rad, D=dist))
