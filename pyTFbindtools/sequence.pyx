cimport cython

import numpy as np
cimport numpy as np

cdef int code_base(unsigned char base):
    if base == 'A':
        return 0
    if base == 'a':
        return 0
    if base == 'C':
        return 1
    if base == 'c':
        return 1
    if base == 'G':
        return 2
    if base == 'g':
        return 2
    if base == 'T':
        return 3
    if base == 't':
        return 3
    return 4

@cython.boundscheck(False)
@cython.wraparound(False)
def code_seq(char* seq):
    cdef int i
    cdef int length = len(seq)
    cdef char coded_base
    cdef np.ndarray[np.int32_t, ndim=2] coded_seq = np.zeros(
        (5, length), dtype=np.int32)
    for i in range(length):
        coded_seq[code_base(seq[i]), i] = 1
    return coded_seq
