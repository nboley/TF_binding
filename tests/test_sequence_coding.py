import os, sys
from pyTFbindtools.sequence import code_seq
import numpy as np

def johnies_code_many_seqs(sequences):
    seq_Strings = []
    for seq in sequences:
        if set(seq) <= set('ACGTacgtN'):
            seq_Strings.append(seq.upper())

    print 'num of sequences: ', len(seq_Strings)
    seq_Lines = []
    for seq in seq_Strings:
        seq_Lines.append(list(seq))

    print 'creating one hot encodings...'
    seq_Array = np.asarray(seq_Lines)
    N, k = np.shape(seq_Array)
    one_hot_encoding = np.zeros((N, 4, k), dtype='bool')
    letters = ['A', 'C', 'G', 'T']
    for i in xrange(len(letters)):
        letter = letters[i]
        one_hot_encoding[:, i, :] = seq_Array == letter

def johnies_code_seq(seq):
    k = len(seq)
    seq_Array = np.asarray(list(seq.upper()))
    one_hot_encoding = np.zeros((4, k), dtype='bool')
    letters = ['A', 'C', 'G', 'T']
    for i in xrange(len(letters)):
        letter = letters[i]
        one_hot_encoding[i, :] = (seq_Array == letter)
    return one_hot_encoding

seq = 'A'*100000
many_seqs = ['A'*100000 for i in xrange(1000)]

def test_code_sequence():
    code_seq(seq)

def test_code_many_seqs():
    johnies_code_many_seqs(many_seqs)

if __name__ == '__main__':
    import timeit
    print timeit.timeit(
        "test_code_sequence()",
        setup="from __main__ import test_code_sequence",
        number=1000)
    print timeit.timeit(
        "test_code_many_seqs()",
        setup="from __main__ import test_code_many_seqs",
        number=1)
