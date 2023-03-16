#!/usr/bin/env python3.10
"""Convert a bin file from FPGA to a fits file

  usage:
    count_from_bin.py <data_log>
    count_from_bin.py (-h|--help)

  options:
    -h --help  show this help message and exit
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import struct
from copy import copy

from imageutils import ext_imagearray2023, convert_e

from scipy.optimize import minimize

thres   = 20
nx      = 144
ny      = 144
nf      = 16

pi  = np.pi
h   = 6.62607015e-34    #Js
c   = 2.99792485e+8     #m/s
k   = 1.380649e-23      #J/K

def makemozaic(im, nx):
  n_sect  = nx*nx
  n_spix  = int((len(im.flatten())/n_sect)**0.5)
  cent    = np.zeros(len(im.flatten())).reshape(np.shape(im))

  for iy in range(nx):
    for ix in range(nx):
      med   = np.median(im[iy*n_spix:(iy+1)*n_spix, ix*n_spix:(ix+1)*n_spix])
      cent[iy*n_spix:(iy+1)*n_spix, ix*n_spix:(ix+1)*n_spix] = np.full((n_spix,n_spix),med)
  return cent

def makemozaic_dev(im, nx):
  n_sect  = nx*nx
  n_spix  = int((len(im.flatten())/n_sect)**0.5)
  cent    = np.zeros(len(im.flatten())).reshape(np.shape(im))

  for iy in range(nx):
    for ix in range(nx):
      region  = im[iy*n_spix:(iy+1)*n_spix, ix*n_spix:(ix+1)*n_spix]
      dev     = (np.median(np.abs(region-np.median(region)))*1.48)/2**0.5
      cent[iy*n_spix:(iy+1)*n_spix, ix*n_spix:(ix+1)*n_spix] = np.full((n_spix,n_spix),dev)
  return cent

def center_hosei(im_ar_sameexp, nx=16):
  im_ar_sameexp = np.array(im_ar_sameexp)
  moza_tot  = []
  
  for im_ar in im_ar_sameexp:
    cent_ar   = []
    for im in im_ar:
      cent    = makemozaic(im, nx=nx)
      cent_ar.append(cent)
    moza_tot.append(cent_ar)

  moza_tot    = np.array(moza_tot)
  gensan_ar   = im_ar_sameexp - moza_tot
  medmoza_ar  = np.mean(moza_tot, axis=0)
  coeff_ar    = np.sqrt(medmoza_ar/moza_tot)
  final_ar    = medmoza_ar  + coeff_ar*gensan_ar
  return final_ar

def cal_image(diff_im, mean_im):
  res_ar  = []
  xs  = 4
  ys  = 4
  n_sect  = xs*ys
  n_spix  = int((len(diff_im[0].flatten())/n_sect)**0.5)

  for im in diff_im: 
    for iy in range(ys):
      for ix in range(xs):
        im_s  = im[iy*n_spix:(iy+1)*n_spix, ix*n_spix:(ix+1)*n_spix]
        #plt.imshow(im_s)
        #plt.show()
        med_s = np.median(im_s)
        var_s = (1.48*np.median(np.abs(im_s - np.median(im_s))))**2.
        #print(med_s, var_s)
        #res_part.append([med_s, var_s])
        res_ar.append([med_s, var_s])
      
  res_ar  = np.array(res_ar).T
  return res_ar

def fname_to_sec(fname):

  h   = float(fname[-10:-8])
  m   = float(fname[-8:-6])
  s   = float(fname[-6:-4])
  t_sec   = h*3600 + m*60 + s

  return t_sec

def phfunc(c, x):
  g   = c[0]
  sr  = c[1]
  al  = c[2]
  be  = c[3]

  R   = 1+4*(al+be)+14*(al+be)**2.-4*al*be
  #sigtot  = 1/(R*g) * x + (sr/g)**2
  sigtot  = 1/g *x + (sr/g)**2

  return sigtot

def linfunc(c, x):
  return c[0]*x+ c[1]


def likelihood(c, args):
  xob   = args[0]
  yob   = args[1]
  zob   = args[2]
  ycal  = phfunc(c, xob)
  #ycal  = linfunc(c, xob)

  return np.sqrt(np.sum((ycal - yob)**2/zob**2))


if __name__ == '__main__':
  args = docopt(__doc__)
  
  print(convert_e(1))
  ofK = 273.5
  logfile   = args["<data_log>"]
  dlist   = np.loadtxt(logfile, delimiter=',',comments='#', dtype='unicode')
  obslist = dlist[(dlist[:,7]=='')]

  res_tot   = []
  image_all = []
  for line in obslist:
    #print(line)
    file_bin  = line[6]
    time      = fname_to_sec(file_bin)
    tmp       = line[5].strip()
    file_flt  = tmp.replace(".","") + "00.dat"
    f_data    = np.loadtxt(file_flt, dtype='f8').T
    Teff      = float(line[1])
    Teff      = Teff + ofK #Kelvin
    dist      = float(line[4])
    rad       = float(line[3])/2. * 1e-3
    texp      = float(line[2])

    #print(file_bin)
    res_ar  = ext_imagearray2023(file_bin)
    image_all.append(res_ar)

  nx  = 8
  #print(np.shape(image_all))
  #himage_all  = center_hosei(image_all, nx=nx)
  phototra  = []
  ph_bin  = []
  for i in range(int(np.shape(image_all)[0]/2)):
    #im1   = himage_all[2*i]
    #im2   = himage_all[2*i+1]
    im1,im2   = center_hosei([image_all[2*i], image_all[2*i-1]], nx=nx)

    im_diff   = im1 - im2
    im_mean   = (im1 + im2)/2.

    for j in range(len(im_diff)):
      
      moz_mean  = makemozaic(im_mean[j], nx=nx)
      moz_diff  = makemozaic_dev(im_diff[j], nx=nx)

      phototra.append([moz_mean, moz_diff**2])
      #plt.scatter(moz_mean, moz_diff**2)
      medx    = np.median(moz_mean)
      medy    = np.median(moz_diff**2)
      err     = np.std(moz_diff**2)
      ph_bin.append([medx,medy,err])


  #print(phototra)
  #exit()
  phototra  = np.hstack(phototra)
  phototra  = np.array((phototra[0].flatten(),phototra[1].flatten()))
  ph_bin    = np.vstack(ph_bin).T
  #print(phototra)
  #exit()
  phototra  = phototra[:,np.argsort(phototra[0])]
  #print(phototra)
  #phototra  = phototra[:,(phototra[0]<38000)]
  ph_bin    = ph_bin[:,ph_bin[0]<40000]
  x0  = [6., 50., 0.0, 0.0]
  #x0  = [0.3, 10000]
  res   = minimize(likelihood, x0, args=ph_bin)
  #print(res)
  param   = res.x
  print("cf, rn: ", param[0], param[1])



  plt.scatter(phototra[0], phototra[1], c="grey", s=1, zorder=1)
  plt.errorbar(ph_bin[0], ph_bin[1], yerr=ph_bin[2], fmt="o", c='orangered', zorder=2)
  plt.plot(phototra[0], phfunc(param, phototra[0]), c='black',zorder=3)
  plt.xlabel("count")
  plt.ylabel("variance")
  #plt.plot(phototra[0], linfunc(param, phototra[0]))
  

  #plt.show()
  plt.savefig("phototrans20230309.png", dpi=200)
  exit()
  