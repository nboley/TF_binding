import os, sys
import math

from itertools import izip, chain

from collections import defaultdict, namedtuple

import numpy as np
np.random.seed(0)

import theano

from scipy.optimize import brentq, approx_fprime

import random

import pyTFbindtools

from pyTFbindtools.motif_tools import (
    load_motifs, logistic, R, T, DeltaDeltaGArray, Motif, load_motif_from_text)
from pyTFbindtools.sequence import code_seq
from log_lhd import calc_log_lhd

PARTITION_FN_SAMPLE_SIZE = 10000

CONVERGENCE_MAX_LHD_CHANGE = None
MAX_NUM_ITER = 10000

CONSTRAIN_MEAN_ENERGY = True
EXPECTED_MEAN_ENERGY = -3.0

CONSTRAIN_BASE_ENERGY_DIFF = True
MAX_BASE_ENERGY_DIFF = 8.0

USE_SHAPE = False

RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
base_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 0: 0, 1: 1, 2: 2, 3: 3}
ShapeData = namedtuple(
    'ShapeData', ['HelT', 'MGW', 'LProT', 'RProT', 'LRoll', 'RRoll'])

def load_shape_data():
    prefix = os.path.join(os.path.dirname(__file__), './shape_data/')
    fivemer_fnames = ["all_fivemers.HelT", "all_fivemers.MGW"]
    fourmer_fnames = ["all_fivemers.ProT", "all_fivemers.Roll"]
    shape_params = defaultdict(list)
    # load shape data for all of the fivemers 
    for fname in chain(fivemer_fnames, fourmer_fnames):
        shape_param_name = fname.split(".")[-1]
        with open(os.path.join(prefix, fname)) as fp:
            for data in fp.read().strip().split(">")[1:]:
                seq, params = data.split()
                param = params.split(";")
                if len(param) == 5:
                    shape_params[seq].append(float(param[2]))
                elif len(param) == 4:
                    shape_params[seq].append(float(param[1]))
                    shape_params[seq].append(float(param[2]))
    
    # cast into a named tuple
    for seq, values in shape_params.iteritems():
        shape_params[seq] = ShapeData(*values)
    
    return dict(shape_params)

shape_data = load_shape_data()

def base_map(base):
    if base == 'N':
        base = random.choice('ACGT')
    return base_map_dict[base]

def enumerate_binding_sites(seq, bs_len):
    for offset in xrange(0, len(seq)-bs_len+1):
        subseq = seq[offset:offset+bs_len].upper()
        yield subseq
        yield "".join(RC_map[base] for base in reversed(subseq))

def iter_fivemers(seq):
    for start in xrange(len(seq) - 5 + 1):
        yield seq[start:start+5]
    return

def est_shape_params_for_subseq(subseq):
    """Est shape params for a subsequence.

    Assumes that the flanking sequence is included, so it returns 
    a vector of length len(subseq) - 2 (because the encoding is done with 
    fivemers)
    """
    res = np.zeros(6*(len(subseq)-4), dtype=theano.config.floatX)
    for i, fivemer in enumerate(iter_fivemers(subseq)):
        if 'N' in fivemer:
            res[6*i:6*(i+1)] = 0
        else:
            res[6*i:6*(i+1)] = shape_data[fivemer]
    return res

def code_subseq(subseq, left_flank_dimer, right_flank_dimer, motif_len):
    """Code a subsequence and it's reverse complement.
    
    """
    if isinstance(subseq, str): subseq = subseq.upper()
    # forward sequence
    len_per_base = 3
    if USE_SHAPE: 
        len_per_base += 6 
    values = np.zeros(motif_len*len_per_base, dtype=theano.config.floatX)
    coded_subseq = np.array([
        pos*3 + (base_map(base) - 1) 
        for pos, base in enumerate(subseq)
        if base_map(base) != 0], dtype=int)
    values[coded_subseq] = 1
    if USE_SHAPE:
        values[3*len(subseq):] = est_shape_params_for_subseq(
            left_flank_dimer + subseq + right_flank_dimer)
    return values


def code_sequence(seq, motif_len,
                  left_flank_dimer="NN", right_flank_dimer="NN"):
    # store all binding sites (subseqs and reverse complements of length 
    # motif )
    coded_bss = []
    seq = left_flank_dimer + seq + right_flank_dimer
    #coded_seq = np.array([base_map[base] for base in seq.upper()])
    for offset in xrange(2, len(seq)-motif_len+1-2):
        subseq = seq[offset:offset+motif_len]
        left_flank = seq[offset-2:offset]
        right_flank = seq[offset+motif_len:offset+motif_len+2]
        values = code_subseq(subseq, left_flank, right_flank, motif_len)
        coded_bss.append(values)

        subseq = "".join(
            RC_map[base] for base in reversed(seq[offset:offset+motif_len]))
        left_flank = "".join(
            RC_map[base] for base in reversed(seq[offset-2:offset]))
        right_flank_flank = "".join(
            RC_map[base] for base in reversed(seq[offset+motif_len:offset+motif_len+2]))
        values = code_subseq(subseq, left_flank, right_flank, motif_len)
        coded_bss.append(values)

    return coded_bss

def code_seqs(seqs, seq_len, n_seqs=None, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    if n_seqs == None: n_seqs = len(seqs)

    # leave 3 rows for each sequence base, and 6 for the shape params
    len_per_base = 3
    if USE_SHAPE:
        len_per_base += 6

    coded_seqs = np.zeros((n_seqs, len_per_base, seq_len), dtype='float32')
    for i, seq in enumerate(seqs):
        # code seq, and then toss the A and N rows
        coded_seq = code_seq(seq)
        # ignore the A and N rows
        coded_seqs[i,:,:] = coded_seq[(1,2,3),:]
    return coded_seqs

def code_RC_seqs(seqs, seq_len, n_seqs=None, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    if n_seqs == None: n_seqs = len(seqs)

    # leave 3 rows for each sequence base, and 6 for the shape params
    len_per_base = 3
    if USE_SHAPE:
        len_per_base += 6

    coded_seqs = np.zeros((n_seqs, len_per_base, seq_len), dtype='float32')
    for i, seq in enumerate(seqs):
        # code seq, and then toss the A and N rows
        coded_seq = code_seq(seq)
        # ignore the A and N rows
        coded_seqs[i,:,:] = np.fliplr(np.flipud(coded_seq[:4,:]))[(1,2,3),:]
    return coded_seqs


    #if ON_GPU:
    #    return theano.shared(coded_seqs)
    #else:
    #    return coded_seqs

def find_consensus_bind_site(seqs, bs_len):
    # produce and initial alignment from the last round
    mers = defaultdict(int)
    for seq in seqs:
        for bs in enumerate_binding_sites(seq, bs_len):
            mers[bs] += 1
    max_cnt = max(mers.values())
    consensus = next(mer for mer, cnt in mers.items() 
                     if cnt == max_cnt)
    return consensus

def find_pwm_from_starting_alignment(seqs, counts):
    assert counts.shape[0] == 4
    bs_len = counts.shape[1]
    
    # code the sequences
    all_binding_sites = []
    all_weights = []
    for seq in seqs:
        binding_sites = list(enumerate_binding_sites(seq, bs_len))
        all_binding_sites.append( binding_sites )
        weights = np.array([1.0/len(binding_sites)]*len(binding_sites))
        all_weights.append(weights)

    # iterate over the alignments
    prev_counts = counts.copy()
    for i in xrange(50):
        # upadte the counts
        for bss, weights in izip(all_binding_sites, all_weights):
            for bs, weight in izip(bss, weights):
                for j, base in enumerate(bs):
                    counts[base_map(base),j] += weight

        # update the weights
        for bss, weights in izip(all_binding_sites, all_weights):
            max_score = 0
            max_pos = None
            for j, bs in enumerate(bss):
                score = sum(
                    counts[base_map(base), k] for k, base in enumerate(bs))
                if score > max_score:
                    max_score = score
                    max_pos = j
                weights[:] = 0
                weights[max_pos] = 1
            weights /= weights.sum()
        if (counts.round() - prev_counts.round() ).sum() == 0: 
            break
        else: 
            prev_counts = counts.copy()
            counts[:,:] = 0

    return (counts/counts.sum(0)).T

def find_pwm(rnds_and_seqs, bs_len):
    consensus = find_consensus_bind_site(
        rnds_and_seqs[max(rnds_and_seqs.keys())], bs_len)
    pyTFbindtools.log("Found consensus %imer '%s'" % (bs_len, consensus))
    counts = np.zeros((4, bs_len))
    for pos, base in enumerate(consensus): 
        counts[base_map(base), pos] += 1000

    # find a pwm using the initial sixmer alignment
    pwm = find_pwm_from_starting_alignment(
        rnds_and_seqs[min(rnds_and_seqs.keys())], counts)
    return pwm

def calc_occ(seq_ddgs, ref_energy, chem_affinity):
    return logistic(-(-chem_affinity+ref_energy+seq_ddgs)/(R*T))

_SelexData = namedtuple('SelexData', ['bg_seqs', 'bnd_seqs'])
class SelexData(_SelexData):
    @property
    def last_rnd_index(self):
        return max(self.bnd_seqs.iterkeys())

    @property
    def first_rnd_index(self):
        return min(self.bnd_seqs.iterkeys())

    @property    
    def last_rnd(self):
        return self.bnd_seqs[self.last_rnd_index]

    @property    
    def first_rnd(self):
        return self.bnd_seqs[self.first_rnd_index]

class PartitionedAndCodedSeqs(object):
    def __init__(self, 
                 rnds_and_seqs, 
                 background_seqs, 
                 use_full_background_for_part_fn=True, 
                 n_partitions=None):
        # set the sequence length
        self.seq_length = len(rnds_and_seqs.values()[0][0])
        self.use_full_background_for_part_fn = use_full_background_for_part_fn
        
        # set the number of data partitions
        if n_partitions is None:
            n_partitions = max(
                5, min(len(seqs)/10000 for seqs in rnds_and_seqs.itervalues()))
        self.n_partitions = n_partitions
        if self.n_partitions <= 3: 
            raise ValueError, "Need at least 3 partitions (test, train, validation)"
        
        # store the full coded sequence arrays
        self.coded_seqs = {}
        for rnd, seqs in rnds_and_seqs.iteritems():
            self.coded_seqs[rnd] = code_seqs(seqs, self.seq_length)
        self.coded_bg_seqs = code_seqs(background_seqs, self.seq_length)

        # store views to partitioned subsets of the data 
        self._partitioned_data = [{} for i in xrange(self.n_partitions)]
        for rnd, coded_seqs in self.coded_seqs.iteritems():
            for partition_index, partition in enumerate(np.array_split(
                    coded_seqs, self.n_partitions)):
                self._partitioned_data[partition_index][rnd] = partition

        # partition the background sequences.
        if self.use_full_background_for_part_fn:
            self._partitioned_bg_data = [
                self.coded_bg_seqs for i in xrange(self.n_partitions)]
        else:
            self._partitioned_bg_data = np.split(
                self.coded_bg_seqs, self.n_partitions)
        
        self.test = SelexData(
            self._partitioned_bg_data[0], self._partitioned_data[0])
        self.validation = SelexData(
            self._partitioned_bg_data[1], self._partitioned_data[1])
        self.train = [
            SelexData(self._partitioned_bg_data[i], self._partitioned_data[i])
            for i in xrange(self.n_partitions-2)
        ]

def estimate_dg_matrix_with_adadelta(
        partitioned_and_coded_rnds_and_seqs,
        init_ddg_array, init_ref_energy,
        dna_conc, prot_conc,
        ftol=1e-12):    
    def calc_penalty(ref_energy, ddg_array):
        penalty = 0
        
        # Penalize models with non-physical mean affinities
        new_mean_energy = ref_energy + ddg_array.sum()/3
        if CONSTRAIN_MEAN_ENERGY:
            penalty += (new_mean_energy - EXPECTED_MEAN_ENERGY)**2

        # Penalize non-physical differences in base affinities
        if CONSTRAIN_BASE_ENERGY_DIFF:
            base_conts = ddg_array.calc_base_contributions()
            energy_diff = base_conts.max(1) - base_conts.min(1)
            penalty += (energy_diff[(energy_diff > 6)]**2).sum()
        #return 0
        return penalty

    def extract_data_from_array(x):
        ref_energy = x[0]
        ddg_array = x[1:].reshape(3,(len(x)-1)/3).astype('float32').view(DeltaDeltaGArray)
        return ref_energy, ddg_array
    
    def f_dg(x, train_index):
        ref_energy, ddg_array = extract_data_from_array(x)
        assert train_index < len(partitioned_and_coded_rnds_and_seqs.train)
        (bg_seqs, bnd_seqs
         ) = partitioned_and_coded_rnds_and_seqs.train[train_index]
        rv = calc_log_lhd(
            ref_energy, 
            ddg_array, 
            bnd_seqs,
            bg_seqs,
            dna_conc, 
            prot_conc)
        penalty = calc_penalty(ref_energy, ddg_array)
        return -rv + penalty

    # ada delta
    validation_lhds = []
    train_lhds = []
    xs = []
    def ada_delta(x0):
        # from http://arxiv.org/pdf/1212.5701.pdf
        e = 1e-6
        p = 0.99
        grad_sq = np.zeros(len(x0))
        delta_x_sq = np.zeros(len(x0))
        
        eps = 1.0
        num_small_decreases = 0
        for i in xrange(MAX_NUM_ITER):
            train_index = random.randrange(
                len(partitioned_and_coded_rnds_and_seqs.train))
            grad = approx_fprime(x0, f_dg, 1e-3, train_index)
            grad_sq = p*grad_sq + (1-p)*(grad**2)
            delta_x = -np.sqrt(delta_x_sq + e)/np.sqrt(
                grad_sq + e)*grad
            delta_x_sq = p*delta_x_sq + (1-p)*(delta_x**2)
            x0 += delta_x.clip(-2, 2) #grad #delta
            train_lhd = -f_dg(x0, train_index)
            validation_lhd = -f_dg(x0, 1)
            ref_energy, ddg_array = extract_data_from_array(x0)

            debug_output = []
            debug_output.append(str(ddg_array.consensus_seq()))
            debug_output.append("Ref: %s" % ref_energy)
            debug_output.append(
                "Mean: %s" % (ref_energy + ddg_array.mean_energy))
            debug_output.append(
                "Min: %s" % ddg_array.calc_min_energy(ref_energy))
            debug_output.append( 
                str(ddg_array.calc_base_contributions().round(2)))
            debug_output.append("Train: %s (%i)" % (train_lhd, train_index))
            debug_output.append("Validation: %s" % validation_lhd)
            debug_output.append("Grad L2 Norm: %.2f" % (
                math.sqrt((grad**2).sum()))
            )
            pyTFbindtools.log("\n".join(debug_output), 'DEBUG')
            
            train_lhds.append(train_lhd)
            validation_lhds.append(validation_lhd)
            xs.append(x0)
            min_iter = 4*len(partitioned_and_coded_rnds_and_seqs.train)
            if i > 2*min_iter and (
                    sum(validation_lhds[-2*min_iter:-min_iter])/min_iter
                    > sum(validation_lhds[-min_iter:])/min_iter ):
                break

        x_hat_index = np.argmax(np.array(validation_lhds))
        return xs[x_hat_index]
    
    bs_len = init_ddg_array.motif_len    

    x0 = init_ddg_array.copy().astype('float32')
    if USE_SHAPE:
        x0 = np.append(x0, np.zeros(6*(len(x0)/3)))
    x0 = np.insert(x0, 0, init_ref_energy)
    x = ada_delta(x0)

    ref_energy, ddg_array = extract_data_from_array(x)
    validation_lhd = calc_log_lhd(
        ref_energy, 
        ddg_array, 
        partitioned_and_coded_rnds_and_seqs.validation.bnd_seqs, 
        partitioned_and_coded_rnds_and_seqs.validation.bg_seqs, 
        dna_conc, 
        prot_conc)

    return ddg_array, ref_energy, validation_lhds, validation_lhd
