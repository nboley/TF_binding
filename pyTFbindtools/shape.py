import os
import random

from itertools import chain, product
from collections import defaultdict, namedtuple

import numpy as np
import theano 

RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
base_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 0: 0, 1: 1, 2: 2, 3: 3}

def base_map(base):
    if base == 'N':
        base = random.choice('ACGT')
    return base_map_dict[base]

def iter_fivemers(seq):
    for start in xrange(len(seq) - 5 + 1):
        yield seq[start:start+5]
    return

ShapeData = namedtuple(
    'ShapeData', ['ProT', 'MGW', 'LHelT', 'RHelT', 'LRoll', 'RRoll'])

fivemer_to_index_map = dict((''.join(fivemer), i) 
                            for (i, fivemer) 
                            in enumerate(sorted(product('ACGT', repeat=5))))
def fivemer_to_index(fivemer):
    return fivemer_to_index_map[fivemer.upper()]

def load_shape_data(center=True):
    prefix = os.path.join(os.path.dirname(__file__), './shape_data/')
    fivemer_fnames = ["all_fivemers.ProT", "all_fivemers.MGW"]
    fourmer_fnames = ["all_fivemers.HelT", "all_fivemers.Roll"]

    # load shape data for all of the fivemers 
    shape_params = np.zeros((4**5, 6))    
    pos = 0
    for fname in chain(fivemer_fnames, fourmer_fnames):
        shape_param_name = fname.split(".")[-1]
        with open(os.path.join(prefix, fname)) as fp:
            for data in fp.read().strip().split(">")[1:]:
                seq, params = data.split()
                param = params.split(";")
                if len(param) == 5:
                    shape_params[fivemer_to_index(seq), pos] = float(param[2])
                elif len(param) == 4:
                    shape_params[fivemer_to_index(seq), pos] = float(param[1])
                    shape_params[fivemer_to_index(seq), pos+1] = float(param[2])
        if fname in fivemer_fnames: pos += 1
        if fname in fourmer_fnames: pos += 2

    if center:
        shape_params = shape_params - shape_params.mean(0)
    return shape_params

def est_shape_params_for_subseq(subseq):
    """Est shape params for a subsequence.

    Assumes that the flanking sequence is included, so it returns 
    a vector of length len(subseq) - 2 (because the encoding is done with 
    fivemers)
    """
    res = np.zeros((6, len(subseq)-4), dtype=theano.config.floatX)
    for i, fivemer in enumerate(iter_fivemers(subseq)):
        fivemer = fivemer.upper()
        if 'AAAAA' == fivemer:
            res[:,i] = 0
        elif 'N' in fivemer:
            res[:,i] = 0
        else:
            res[:,i] = shape_data[fivemer_to_index(fivemer)]
    return res

def code_sequence_shape(seq, left_flank_dimer="NN", right_flank_dimer="NN"):
    full_seq = left_flank_dimer + seq + right_flank_dimer
    return est_shape_params_for_subseq(full_seq)

def code_seqs_shape_features(seqs, seq_len, n_seqs):
    shape_features = np.zeros(
        (n_seqs, 6, seq_len), dtype=theano.config.floatX)
    RC_shape_features = np.zeros(
        (n_seqs, 6, seq_len), dtype=theano.config.floatX)
    
    for i, seq in enumerate(seqs):
        shape_features[i, :, :] = code_sequence_shape(seq)
        shape_features[i, :, :] = code_sequence_shape(
            "".join(RC_map[base] for base in seq[::-1]))

    return shape_features, RC_shape_features

shape_data = load_shape_data()
