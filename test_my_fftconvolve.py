import os, sys

import numpy as np
from scipy.signal import fftconvolve

from scipy.fftpack import rfftn, irfftn

USE_FFT = True

def my_convolve(in1, in2):
    in1 = asarray(in1)
    in2 = asarray(in2)
    s1 = array(in1.shape)
    s2 = array(in2.shape)
    shape = s1 + s2 - 1
    fshape = [_next_regular(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    ret = irfftn(rfftn(in1, fshape) *
                 rfftn(in2, fshape), fshape)[fslice].copy()
    return ret

def t1():
    x = np.arange(100000, dtype=float)
    x = x/x.sum()
    xnew = np.arange(1000, dtype=float)
    xnew = xnew/xnew.sum()
    for j in xrange(40):
        if USE_FFT:
            x = fftconvolve(x, xnew)
        else:
            x = np.convolve(x, xnew)
    return x

res = t1()
print res.sum(), res.shape
