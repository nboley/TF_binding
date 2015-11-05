import cython
from cpython.string cimport PyString_AsString
from cython.parallel import prange
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free, realloc

import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DEF NUM_BASES = 4 

################################################################################
# Build 'base_prbs' lookup table mapping DNA characters to one hot encoding prbs
# The lookup table is length 256*4, where each 4 entry stores the 'A' 
# contribution. So, for example, and 'A' has an ascii code 65, and so
# the 4*65, 4*65+1, 4*65+2, 4*65+3 entries are (1.0, 0.0, 0.0, 0.0). Similarly 
# a 'k (ascii code 107) indicates either a G or T, and so entries 
# 4*107+(0,1,2,3) are (0, 0, 0.5, 0.5).

encoding = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'K': [0, 0, 0.5, 0.5],
    'M': [0.5, 0.5, 0, 0],
    'R': [0.5, 0, 0.5, 0],
    'Y': [0, 0.5, 0, 0.5],
    'S': [0, 0.5, 0.5, 0],
    'W': [0.5, 0, 0, 0.5],
    'B': [0, 1./3, 1./3, 1./3],
    'V': [1./3, 1./3, 1./3, 0],
    'H': [1./3, 1./3, 0, 1./3],
    'D': [1./3, 0, 1./3, 1./3],
    'X': [0.25, 0.25, 0.25, 0.25],
    'N': [0.25, 0.25, 0.25, 0.25]
}
for base in list(encoding.keys()):
    encoding[base.lower()] = encoding[base]

cdef DTYPE_t *base_prbs = [0]*(256*4)
for character_code in range(256):
    values = ( encoding[chr(character_code)] 
               if chr(character_code) in encoding 
               else [0.0, 0.0, 0.0, 0.0] )
    for offset, value in enumerate(values):
        base_prbs[character_code*4 + offset] = value

# END build lookup table
################################################################################

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
    cdef np.ndarray[np.int32_t, ndim=2] coded_seq = np.zeros((5, length), dtype=np.int32)
    #coded_seq = np.zeros((5,len(seq)))
    for i in range(length):
        coded_seq[code_base(seq[i]), i] = 1
    return coded_seq

@cython.boundscheck(False)
def one_hot_encode_sequences_FORLOOP(sequences):
    cdef int position, sequence_index, encoding_index
    cdef int num_sequences = len(sequences)
    cdef int sequence_length = len(sequences[0])
    cdef char *sequence
    cdef np.ndarray[DTYPE_t, ndim=3] encoded_sequences = \
        np.empty((num_sequences, sequence_length, NUM_BASES), DTYPE)
    for sequence_index in range(num_sequences):
        sequence = PyString_AsString(sequences[sequence_index])
        for position in range(sequence_length):
            for encoding_index in range(NUM_BASES):
                encoded_sequences[sequence_index, position, encoding_index] = \
                base_prbs[sequence[position] * NUM_BASES + encoding_index]
    return encoded_sequences

@cython.boundscheck(False)
cdef int one_hot_encode_c_sequences_MEMCPY(char** sequences, 
                                           int num_sequences,
                                           int sequence_length,
                                           char* encoded_sequences):
    cdef char* sequence
    cdef char base
    cdef int position, sequence_index
    for sequence_index in range(num_sequences):
        sequence = sequences[sequence_index]
        for position in range(sequence_length):
            base = sequence[position]
            # if we reach a null, then we have exhausted this string
            # so break
            if base == 0: break
            memcpy( 
                <DTYPE_t*> encoded_sequences 
                    + sequence_index*sequence_length*NUM_BASES
                    + position*NUM_BASES, 
                <DTYPE_t*> base_prbs 
                    + base*NUM_BASES,
                NUM_BASES*sizeof(DTYPE_t)
            )
    return 0

cdef object convert_py_string_to_c_string(
        object sequences, char*** c_sequences):
    DEF NUM_SEQ_STEP_SIZE = 15
    
    cdef int num_alld_seqs = NUM_SEQ_STEP_SIZE
    c_sequences[0] = <char**> malloc(num_alld_seqs * sizeof(char*))
    
    cdef int max_sequence_length = 0
    cdef int sequence_index = 0
    for sequence in sequences:
        if max_sequence_length < len(sequence):
            max_sequence_length = len(sequence)
        c_sequences[0][sequence_index] = PyString_AsString(
            sequence)
        sequence_index += 1
        # if we have run out of space and need to reallocate
        if sequence_index == num_alld_seqs:
            num_alld_seqs += NUM_SEQ_STEP_SIZE
            c_sequences[0] = <char**> realloc(
                c_sequences[0], num_alld_seqs*sizeof(char*))

    # reduce the allocation size
    num_alld_seqs = sequence_index
    c_sequences[0] = <char**> realloc(
        c_sequences[0], num_alld_seqs*sizeof(char*))
    return num_alld_seqs, max_sequence_length

def one_hot_encode_sequences(sequences):
    cdef char** c_sequences = NULL;
    cdef int seq_length, num_seqs
    cdef np.ndarray[DTYPE_t, ndim=3] encoded_sequences
    
    try:
       # convert the python strings to c_strings. 
        num_seqs, seq_length = convert_py_string_to_c_string(
            sequences, &c_sequences)
        
        # allocate an numpy array to store the encoded sequences in
        encoded_sequences = np.empty(
            (num_seqs, seq_length, NUM_BASES), dtype=DTYPE)
        res = one_hot_encode_c_sequences_MEMCPY(
            c_sequences, num_seqs, seq_length, encoded_sequences.data)
        return encoded_sequences
    finally:
        free(c_sequences)

def profile( seq_len, n_seq, n_test_iterations ):
    """Compare the speed of the two one-hot-encoding implementations.

    To use this from the command line run:
    python -c "import pyximport; pyximport.install(); import test; test.profile(200000, 1000, 1)"
    """
    import timeit
    sequence = 'A'*seq_len
    sequences = [sequence for x in xrange(n_seq)]

    t_FORLOOP = timeit.Timer(
        lambda: one_hot_encode_sequences_FORLOOP(sequences) )
    t_MEMCPY = timeit.Timer(
        lambda: one_hot_encode_sequences(sequences) )
    print "ForLoop:", t_FORLOOP.timeit(number=n_test_iterations)
    print "MemCpy :", t_MEMCPY.timeit(number=n_test_iterations)
    return 

def test_implementations(n_tests=1000):
    import random
    def sample_random_seqs(n_sims, seq_len):
        return ["".join(random.choice('ACGT') for j in xrange(seq_len))
                for i in xrange(n_sims)]

    for i in xrange(n_tests):
        for seq_len in (10, 100, 1000, 10000):
            seqs = sample_random_seqs(10, seq_len)
            code_1 = one_hot_encode_sequences(seqs)
            code_2 = one_hot_encode_sequences_FORLOOP(seqs)
            assert (np.abs(code_1 - code_2)).sum() < 1e-6
    print "PASS"
    return
