import numpy as np

def code_base(base):
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

def code_seq(seq):
    coded_seq = np.zeros((5,len(seq)), dtype=np.int)
    print coded_seq.shape
    print coded_seq[0,0]
    for i, base in enumerate(seq):
        coded_base = code_base(base)
        coded_seq[i, coded_base] = 1
    return coded_seq
