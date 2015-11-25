import os, sys
import random
import math

import numpy as np
from scipy.signal import convolve, fftconvolve

from numpy.fft import rfft, rfftn, irfft, irfftn
#from scipy.fftpack import rfft

from pyTFbindtools.sequence import one_hot_encode_sequence

USE_MY_FFT = False

def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

def my_convolve(in1, in2, mode='valid'):
    assert mode == 'valid'
    shape = len(in1) + len(in2) - 1
    fshape = _next_regular(shape)
    ret = irfft(rfft(in1, fshape) *
                rfft(in2, fshape), fshape)
    return ret[:shape]

def multi_convolve(signals):
    res = signals[0]
    for signal in signals[1:]:
        res = fftconvolve(res, signal)
    return res

def my_multi_convolve(signals):
    shape = sum(len(x)-1 for x in signals) + 1
    fshape = _next_regular(shape)
    res = rfft(signals[0], fshape)
    for signal in signals[1:]:
        res *= rfft(signal, fshape)
    ret = irfft(res, fshape)
    return ret[:shape]

def t1():
    xs = []
    for i in xrange(50):
        x = np.arange(1000, dtype=float)
        x = x/x.sum()
        xs.append(x)

    for j in xrange(30):
        if USE_MY_FFT:
            my_multi_convolve(xs)
        else:
            multi_convolve(xs)
    return

def overlap_add_1D(x, h, mode='full'):
    # Evaluate the best value of N and L (L>0, N = M+L-1 nearest to power of 2).
    x_len = len(x)
    # pad x so that the boundaries are dealt with correctly
    x = np.pad(x, (0,len(h)), 'constant', constant_values=0)
    block_size = max(2**8, len(h))

    # make sure that the desired block size is long enough to capture the motif
    N = int(2**math.ceil(np.log2(block_size+len(h)-1)))
    step_size = N-len(h)+1
    
    H = rfft(h,N)
    
    n_blocks = int(math.ceil(float(len(x))/step_size))
    y = np.zeros((n_blocks+1)*step_size)
    for block_index in xrange(n_blocks):
        start = block_index*step_size
        yt = irfft( rfft(x[start:start+step_size],N) * H, N)
        #print "Yt", yt #[overlap:20], yt[-10:]
        y[start:start+N] += yt

    y = y[:len(h)+x_len-1]
    if mode == 'full':
        return y
    elif mode == 'valid':
        return y[len(h)-1:-len(h)+1]
    elif mode == 'same':
        raise NotImplementedError, "'same' mode is not implemented"

def overlap_add_multi_channel(x, h, mode='full', block_power=11):
    # pad x so that the boundaries are dealt with correctly
    x_len = x.shape[0]
    num_channels = h.shape[1]
    h_len = h.shape[0]
    assert x.shape[1] == num_channels
    
    x = np.vstack((np.zeros((h_len, num_channels)), 
                   x, 
                   np.zeros((h_len, num_channels))))    

    # make sure that the desired block size is long enough to capture the motif
    block_size = max(2**block_power, h_len)
    N = int(2**math.ceil(np.log2(block_size+h_len-1)))
    step_size = N-h_len+1
    
    H = rfftn(h,(N,num_channels))
    n_blocks = int(math.ceil(float(len(x))/step_size))
    y = np.zeros((n_blocks+1)*step_size)
    for block_index in xrange(n_blocks):
        start = block_index*step_size
        yt = irfftn( rfftn(x[start:start+step_size,:],(N, num_channels))*H, 
                     (N, num_channels) )
        y[start:start+N] += yt[:,num_channels-1]

    y = y[h_len:2*h_len+x_len-1]
    if mode == 'full':
        return y
    elif mode == 'valid':
        return y[h_len-1:-h_len+1]
    elif mode == 'same':
        raise NotImplementedError, "'same' mode is not implemented"

def test_convolve_implementations(seq_len=1000, motif_len=1200):
    def test(x, f):
        res1 = overlap_add_multi_channel(x, f)
        res2 = convolve(x, f)[:,3]
        #res3 = my_convolve(x, f)
        print np.abs(res1 - res2).sum()
        assert np.abs(res1 - res2).sum() < 1e-6
        #assert np.abs(res2 - res3).sum() < 1e-6
        #assert np.abs(res1 - res3).sum() < 1e-6

    seq = "".join(random.choice('ACGT') for j in xrange(seq_len))
    coded_seq = one_hot_encode_sequence(seq)[0]
    motif = np.random.rand(motif_len, 4)
    print coded_seq.shape
    print motif.shape
    test(coded_seq, motif)

def test_motif_scoring(seq_len=10*1, motif_len=5):
    #convolve_fn = fftconvolve
    seq = "".join(random.choice('ACGT') for j in xrange(seq_len))
    #seq = 'A'*seq_len
    coded_seq = one_hot_encode_sequence(seq)[0]
    motif = np.random.rand(motif_len,4)
    #motif = np.ones((motif_len, 4))
    print convolve(coded_seq, motif, mode='valid')[:,0]
    print overlap_add_multi_channel(coded_seq, motif, mode='valid')
    return
    mode = 'full'
    print "new        :", overlap_add(coded_seq, motif, mode=mode)
    print "scipy      :", convolve(coded_seq, motif, mode=mode)
    print "my_convolve:", my_convolve(coded_seq, motif)
    return
    fshape = _next_regular(coded_seq.shape[0])

    coded_seq_freq = rfft(coded_seq, n=fshape)
    print coded_seq.shape
    print coded_seq_freq.shape
    print motif.shape
    motif_freq = rfft(motif, n=fshape)
    print motif_freq
    print motif_freq.shape
    return

    print coded_seq.shape
    print motif.shape
    for i in xrange(1000):
        res = convolve_fn(coded_seq, motif, mode='valid').shape
    pass

seq_len = 10000000
motif_len = 20
seq = "A"*seq_len
coded_seq = one_hot_encode_sequence(seq)[0]
motif = np.random.rand(motif_len,4)

def test(block_power):
    overlap_add_multi_channel(coded_seq, motif, 'valid', block_power)

def test2():
    convolve(coded_seq, motif, mode='valid')

def test3():
    fftconvolve(coded_seq, motif, mode='valid')

def main():
    import timeit
    for i in xrange(8, 14):
        print i, timeit.timeit("test(%i)" % i, setup="from __main__ import test", number=1)
    #print "SciPY", timeit.timeit("test2()", setup="from __main__ import test2", number=1)
    #print timeit.timeit("test3()", setup="from __main__ import test3", number=1)
    return

    #x = np.ones(10, dtype=float)
    #f = np.ones(5, dtype=float)
    #x = np.hstack((np.zeros(10), x, np.zeros(10)))
    #N = len(x)+len(f)-1
    #print irfft(rfft(x, N)*rfft(f, N))
    #test_motif_scoring()
    #return
    
    for i in xrange(1000):
        test_convolve_implementations()
    return
    ## timing
    r_len = 1000
    x = np.arange(r_len, dtype=float)
    x = x/x.sum()
    xnew = np.arange(100, dtype=float)
    xnew = xnew/xnew.sum()
    print np.convolve(r_len, xnew)
    print fftconvolve(r_len, xnew)
    print my_convolve(r_len, xnew)
    pass


main()
