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


def obs_nphoton(Teff, lamb, filter, R=0.015, D=0.1):

    #qe          = 0.80 #
    qe          = 1.00 #
    size_pix    = 15e-6 #m

    I   = planck_func(Teff, lamb)
    irr = flux_furnace(D,R, I)
    obs = irr*qe*filter

    rx   = np.linspace(0.0001,0.01,100)
    #plt.plot(rx, flux_furnace(1,rx/2.,1))
    #plt.show()
    #exit()

    #plt.yscale("log")
    #plt.xscale("log")
    #plt.plot(lamb, planck_func(800+273,lamb))
    #plt.xlim((1e-6,1.7e-6))
    #plt.plot(lamb*1e9, planck_func(1400+273,lamb)*1e-3*1e-6)
    #plt.plot(lamb, irr)
    #plt.show()
    #exit()
    dw  = np.median(lamb[1:] - lamb[:-1])

    e_photo = h*c/lamb
    #plt.plot(lamb, I*filter/e_photo )
    #print(np.sum(I*filter*dw/e_photo))
    #plt.plot(lamb, I*filter)
    #plt.plot(lamb, I)
    #plt.scatter(np.arange(len(dw)), dw)
    #plt.show()
    #exit()
    dp  = obs[1:]*dw/e_photo[1:]
    n_s_m2  = np.sum(dp) #s/m^2
    n_s     = n_s_m2*(size_pix**2)
    #J/s = W

    return n_s


def plotfunc(x0,y0, x1, y1):
    fig     = plt.figure(figsize=(7,3))
    plt.rcParams["font.family"] = "Arial"
    ax1     = fig.add_subplot(1,2,1)
    ax2     = fig.add_subplot(1,2,2)

    ax1.set_title("separation: 50 cm")
    ax2.set_title("T: 1273.15 K (1000 C$^\circ$)")

    ax1.set_ylabel("$N/s$ [x $10^{10}$]")
    ax1.set_xlabel("$T_{eff}$ [K]")
    ax2.set_xlabel("separation [cm]")
    ax1.plot(x0,y0/1e10, lw=1, c='black') #uW
    ax2.plot(x1*100,y1/1e10, lw=1, c='black') #uW
    ax2.set_yscale('log')

    plt.tight_layout()
    #plt.show()
    plt.savefig("plot.png", dpi=200)


from scipy.interpolate import interp1d
def filter(filter_array, x):
    wav = filter_array[0]*1e-9
    trs = filter_array[1]*1e-2
    #f   = interp1d(wav, trs, kind="cubic")
    f   = interp1d(wav, trs, kind="linear")
    x_narrow    = x[((wav[-1] < x)&(x < wav[0]))]
    return f(x_narrow), x_narrow

eV_J    = 1.60218e-19
if __name__=='__main__':

    ofK     = 273.15

    f_file  = sys.argv[1]
    f_data  = np.loadtxt(f_file, dtype='f8').T

    #T       = 1400. + ofK
    #D_ar    = np.array([0.7, 0.8, 0.9, 1., 1.1, 1.2,1.3])
    D_ar    = np.array([0.7, 1.0, 1.3, 1.6, 1.9, 2.2])
    D       = 1.0
    D_ar    = D_ar[::-1]
    T_ar    = np.array([800, 1000, 1200, 1400]) + ofK
    #R_ar    = np.array([5e-4, 1e-3, 1.5e-3, 3.e-3, 5.e-3, 7e-3, 1.e-2, 1.5e-2])
    #R_ar    = np.array([1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3.e-3, 5.e-3])
    #R_ar    = np.array([5e-5, 1e-4, 1.5e-4, 2.e-4, 2.5e-4, 3e-4, 3.5e-4])
    R_ar    = np.array([5e-5, 2.5e-4, 5e-4])
    #DD,RR   = np.meshgrid(D_ar, R_ar)
    TT,RR   = np.meshgrid(T_ar, R_ar)

    n_vsT   = []
    n_vsD   = []
    lamb    = np.linspace(0.8e-6, 1.8e-6, 1000)#m

    f,x     = filter(f_data, lamb)

    c_ar    = []
    i   = 0
    ind = []
    #for D in D_ar:
    for T in T_ar:
        for R in R_ar:
            ind.append(i)
            i+=1
            c_ar.append(obs_nphoton(T, x, f, R=R, D=D))
    c_ar    = np.array(c_ar)
    c_ar    = c_ar.reshape((len(T_ar),len(R_ar)))
    ind     = np.array(ind)
    ind     = ind.reshape((len(T_ar),len(R_ar)))
    norm    = simple_norm(c_ar, 'log', percent=99.)

    plt.figure(figsize=(6,6))
    plt.imshow(c_ar, norm=norm, cmap='rainbow')
    plt.xticks(np.arange(len(R_ar)), R_ar*2)
    plt.yticks(np.arange(len(T_ar)), T_ar)
    for y in range(len(T_ar)):
        for x in range(len(R_ar)):
            plt.text(x,y ,"{:.1e}".format(c_ar[y,x]),ha='center',va='center', size=8)
    #plt.savefig("count_map_zoom.png", dpi=200)
    plt.show()
    #plt.savefig("count_map.png", dpi=200)
    exit()
