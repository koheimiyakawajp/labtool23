#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import struct
from cal_nphoton_map import obs_nphoton, filter
from scipy.interpolate import interp1d
from copy import copy

def ext_imagearray2303(file_bin, flg=1, ndummy=8):

  with open(file_bin, mode='rb') as f:
    # 1 frame is 2 * 2**15 bytes, but only 2*144*144 bytes contain data
    data_b = f.read(2*nx*ny)
    f.read(2*2**15 - 2*nx*ny)
    for i in range(nf-1):
      data_b = data_b + f.read(2*nx*ny)
      f.read(2**16 - 2*nx*ny)

    # Change little endian to big endian
    # 符号とかは途中
    # H: unsigned short
    # h: short

    data = struct.unpack('<'+str(nx*ny*16)+'h', data_b)  # H: signed short
    data = [i //2  for i in data]
    res_ar  = []

    frame_0 = np.array(data[0:nx*ny], dtype='i8').reshape(nx,ny)
    frame_0 = frame_0[ndummy:-1*ndummy,ndummy:-1*ndummy]
    frame_a = np.array(data[nx*ny:2*nx*ny], dtype='i8').reshape(nx,ny)
    frame_a = frame_a[ndummy:-1*ndummy,ndummy:-1*ndummy] - frame_0

    im_ar   = []
    imsum = copy(frame_a)
    for i in range(nf-2):
      frame_b   = np.array(data[(i+2)*nx*ny:(i+3)*nx*ny], dtype='i8').reshape(nx,ny)
      frame_b   = frame_b[ndummy:-1*ndummy,ndummy:-1*ndummy]  - frame_0

      frame_d   = frame_b - frame_a
      frame_d[(frame_d<-2**14)]   += 2**15
      imsum     += frame_d
      frame_a   = frame_b
      im_ar.append(copy(imsum))

  return np.array(im_ar)

def convert_e(count):
  CE  = 4.5*10**(-6) #V/e-
  amp = (200/18 + 1)*(15/56 +1)
  vper_c  = 20/(2**16) #V
  c_rate   =  vper_c/(CE*amp)
  
  electron  = count*c_rate
  return electron