import os, sys
import math
import random

from itertools import izip, chain

from collections import defaultdict, namedtuple

import numpy as np
np.random.seed(0)

from scipy.optimize import brentq, approx_fprime
from scipy.stats import ttest_ind

import theano

import pyTFbindtools

from pyTFbindtools.motif_tools import (
    load_motifs, logistic, R, T, DeltaDeltaGArray, Motif, load_motif_from_text)
from pyTFbindtools.sequence import one_hot_encode_sequences, code_seq
from log_lhd import calc_log_lhd, calc_binding_site_energies

PARTITION_FN_SAMPLE_SIZE = 10000

MAX_NUM_ITER = 10000

CONSTRAIN_MEAN_ENERGY = True
EXPECTED_MEAN_ENERGY = -3.0

CONSTRAIN_BASE_ENERGY_DIFF = True
MAX_BASE_ENERGY_DIFF = 8.0

USE_SHAPE = False

def code_seqs(seqs, seq_len, n_seqs=None, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    if n_seqs == None: n_seqs = len(seqs)

    coded_seqs = one_hot_encode_sequences(seqs)
    # remove the a's row
    coded_seqs = coded_seqs[:,:,(1,2,3)]
    assert coded_seqs.shape[1] == seq_len
    # swap the axes so that the array is in sequence-base-position order
    coded_seqs = np.swapaxes(coded_seqs, 1, 2)
    # return a copy to make sure that we're not passing a (slow) view around 
    return coded_seqs.copy()

def enumerate_binding_sites(seq, bs_len):
    for offset in xrange(0, len(seq)-bs_len+1):
        subseq = seq[offset:offset+bs_len].upper()
        yield subseq
        yield "".join(RC_map[base] for base in reversed(subseq))

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

def sample_random_seqs(n_sims, seq_len):
    return ["".join(random.choice('ACGT') for j in xrange(seq_len))
            for i in xrange(n_sims)]

def find_best_shift(coded_seqs, ddg_array, coded_bg_seqs=None):
    def find_flanking_base_cnts(seqs, base_offset):
        # calculate the ddg energies for all binding sites. We use this to align
        # the binding sities within the sequences
        # find the index of the best offset
        energies = calc_binding_site_energies(seqs, ddg_array)
        best_offsets = np.argmin(energies, 1)

        ## find the binding sites which align to the sequence boundary. We must 
        ## remove these because the flanking sequence is not random, and so it
        ## will bias the results
        # deal with left shifts
        if base_offset > 0:
            base_offset += (ddg_array.motif_len - 1)
        base_indices = best_offsets + base_offset
        # find which offsets fit inside of the random portion of the sequence
        valid_indices = (base_indices > 0)&(base_indices < seq_len)
        base_cnts = seqs[
            valid_indices,:,base_indices[valid_indices]].sum(0)
        # add in the A counts
        base_cnts = np.insert(
            base_cnts, 0, valid_indices.sum()-base_cnts.sum())
        return base_cnts

    def calc_entropy(cnts):
        if len(cnts.shape) == 1: cnts = cnts[None,:]
        ps = np.array(cnts+1, dtype=float)/(cnts.sum(1)+4)[:,None]
        return -(ps*np.log(ps+1e-6)).sum(1)
    
    def estimate_entropy_significance(cnts, bg_cnts, n_samples=100000):
        bg_samples = np.random.multinomial(
            bg_cnts.sum(), (1.0+bg_cnts)/(bg_cnts.sum()+4.0), n_samples)
        obs_samples = np.random.multinomial(
            cnts.sum(), (1.0+cnts)/(cnts.sum()+4.0), n_samples)
        stat, p_value = ttest_ind(
            calc_entropy(bg_samples),
            calc_entropy(obs_samples)) 
        return stat, p_value

    seq_len = coded_seqs.shape[2]
    if coded_bg_seqs == None:
        coded_bg_seqs = sample_random_coded_seqs(coded_seqs.shape[0], seq_len)

    ### Code to downsample - I dont think that we want this but the 
    ### entropy calculation isn't perfect so Ill leave it
    #sample_size = min(coded_seqs.shape[0], coded_bg_seqs.shape[0])
    #coded_seqs = coded_seqs[
    #    np.random.choice(coded_seqs.shape[0], sample_size),:,:]
    #coded_bg_seqs = coded_bg_seqs[
    #    np.random.choice(coded_bg_seqs.shape[0], sample_size),:,:]
    
    entropies_and_offsets = []
    for offset in chain(xrange(max(-3, -seq_len+ddg_array.motif_len+1), 0), 
                        xrange(1,min(4, seq_len-ddg_array.motif_len))):
        cnts = find_flanking_base_cnts(coded_seqs, offset)
        bg_cnts = find_flanking_base_cnts(coded_bg_seqs, offset)
        t_stat, p_value = estimate_entropy_significance(cnts, bg_cnts)
        entropies_and_offsets.append(
            (-t_stat, 
             p_value, 
             offset, 
             calc_entropy(cnts)[0], 
             calc_entropy(bg_cnts)[0]))
    entropies_and_offsets.sort()

    pyTFbindtools.log("Shift\tentropy\tbg entropy:", 'VERBOSE')
    for _, _, shift, bg_entropy, entropy in entropies_and_offsets:
        pyTFbindtools.log(
            "%i\t%.2e\t%.2e" % (shift, entropy, bg_entropy), 'VERBOSE')
    
    if len(entropies_and_offsets) == 0:
        return None

    if entropies_and_offsets[0][0] > 0 or entropies_and_offsets[0][1] > 0.05:
        return None
    
    if entropies_and_offsets[0][2] > 0:
        return 'RIGHT'
    return 'LEFT'

FitSelexModel = namedtuple("FitSelexModel", [
    'ddg_array', 'ref_energy', 
    'lhd_path', 'lhd_hat', 
    'prev_validation_lhd', 'new_validation_lhd']
)

def progressively_fit_model(
        rnds_and_seqs, background_seqs, 
        ddg_array, ref_energy, 
        dna_conc, prot_conc,
        partition_background_seqs):
    pyTFbindtools.log("Coding sequences", 'VERBOSE')
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs, 
        background_seqs, 
        use_full_background_for_part_fn=(not partition_background_seqs)
    )
    
    while True:
        bs_len = ddg_array.motif_len
        prev_validation_lhd = calc_log_lhd(
            ref_energy, ddg_array, 
            partitioned_and_coded_rnds_and_seqs.validation.bnd_seqs,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs,
            dna_conc, prot_conc)
        pyTFbindtools.log("Starting lhd: %.2f" % prev_validation_lhd, 'VERBOSE')
        
        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, lhd_path, lhd_hat 
            ) = estimate_dg_matrix_with_adadelta(
                partitioned_and_coded_rnds_and_seqs,
                ddg_array, ref_energy,
                dna_conc, prot_conc)

        pyTFbindtools.log(ddg_array.consensus_seq(), 'VERBOSE')
        pyTFbindtools.log("Ref: %s" % ref_energy, 'VERBOSE')
        pyTFbindtools.log(
            "Mean: %s" % (ref_energy + ddg_array.sum()/3), 'VERBOSE')
        pyTFbindtools.log(
            "Min: %s" % ddg_array.calc_min_energy(ref_energy), 'VERBOSE')
        pyTFbindtools.log(
            str(ddg_array.calc_base_contributions().round(2)), 'VERBOSE')

        new_validation_lhd = calc_log_lhd(
            ref_energy, 
            ddg_array, 
            partitioned_and_coded_rnds_and_seqs.validation.bnd_seqs,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs,
            dna_conc,
            prot_conc)

        pyTFbindtools.log("Prev: %.2f\tCurr: %.2f\tDiff: %.2f" % (
            prev_validation_lhd, 
            new_validation_lhd, 
            new_validation_lhd-prev_validation_lhd
        ), 'VERBOSE')

        yield FitSelexModel( ddg_array, 
                             ref_energy, 
                             lhd_path, 
                             lhd_hat, 
                             prev_validation_lhd, 
                             new_validation_lhd)
        
        if ( bs_len >= 20 
             or bs_len+1 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break
        
        pyTFbindtools.log("Finding best shift", 'VERBOSE')
        shift_type = find_best_shift(
            partitioned_and_coded_rnds_and_seqs.validation.last_rnd,
            ddg_array,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs)
        if shift_type == None:
            pyTFbindtools.log("Model has finished fitting", 'VERBOSE')
            break
            
        if shift_type == 'LEFT':
            pyTFbindtools.log("Adding left base to motif", level='VERBOSE' )
            ddg_array = np.hstack((np.zeros((3,1), dtype='float32'), ddg_array)
                              ).view(DeltaDeltaGArray)
        elif shift_type == 'RIGHT':
            pyTFbindtools.log("Adding right base to motif", level='VERBOSE' )
            ddg_array = np.hstack((ddg_array, np.zeros((3,1), dtype='float32'))).view(
                DeltaDeltaGArray)
        else:
            assert False, "Unrecognized shift type '%s'" % shift_type
        ref_energy = ref_energy
        
    return
