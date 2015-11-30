import os, sys
import math
import random

from itertools import izip, chain

from collections import defaultdict, namedtuple

import numpy as np
np.set_printoptions(precision=2)
np.random.seed(0)

from scipy.optimize import brentq, approx_fprime
from scipy.stats import ttest_ind

import theano

from pyDNAbinding.binding_model import (
    FixedLengthDNASequences, est_chem_potential, 
    EnergeticDNABindingModel, DeltaDeltaGArray )
from pyDNAbinding.sequence import OneHotCodedDNASeq

import pyTFbindtools

from pyTFbindtools.motif_tools import (
    load_motifs, logistic, R, T,
    Motif, load_motif_from_text)
import pyTFbindtools.sequence
from pyTFbindtools.shape import code_seqs_shape_features
import log_lhd

MAX_NUM_ITER = 10000

CONSTRAIN_MEAN_ENERGY = True
EXPECTED_MEAN_ENERGY = -3.0

CONSTRAIN_BASE_ENERGY_DIFF = True
MAX_BASE_ENERGY_DIFF = 6.0

USE_SHAPE = False

MAX_BS_LEN = 18

RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
base_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 0: 0, 1: 1, 2: 2, 3: 3}

def base_map(base):
    if base == 'N':
        base = random.choice('ACGT')
    return base_map_dict[base]

def one_hot_encode_sequences(seqs, n_seqs=None, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    if n_seqs == None: n_seqs = len(seqs)

    coded_seqs = pyTFbindtools.sequence.one_hot_encode_sequences(seqs)
    # remove the a's row
    #coded_seqs = coded_seqs[:,:,(1,2,3)]
    # swap the axes so that the array is in sequence-base-position order
    # return a copy to make sure that we're not passing a (slow) view around 
    #coded_seqs = np.swapaxes(coded_seqs, 1, 2).copy()
    return coded_seqs.copy()

class CodedSeqs(object):
    def _array_split_or_none(self, array, n_partitions):
        if array is None: 
            return [None]*n_partitions
        return np.array_split(array, n_partitions)

    def iter_one_hot_coded_seqs(self):
        for seq in self.one_hot_coded_seqs:
            yield seq.view(OneHotCodedDNASeq)

    def __init__(self, 
                 one_hot_coded_seqs, 
                 shape_coded_fwd_seqs=None, 
                 shape_coded_RC_seqs=None):
        self.n_seqs = one_hot_coded_seqs.shape[0]
        self.seq_length = one_hot_coded_seqs.shape[1]

        self.one_hot_coded_seqs = one_hot_coded_seqs
        self.shape_coded_fwd_seqs = shape_coded_fwd_seqs
        self.shape_coded_RC_seqs = shape_coded_RC_seqs

    def iter_coded_seq_splits(self, n_partitions):
        for (one_hot_seqs, fwd_shape_seqs, RC_shape_seqs
            ) in izip(self._array_split_or_none(self.one_hot_coded_seqs, n_partitions),
                      self._array_split_or_none(self.shape_coded_fwd_seqs, n_partitions),
                      self._array_split_or_none(self.shape_coded_RC_seqs, n_partitions)
                ):
            yield CodedSeqs(one_hot_seqs, fwd_shape_seqs, RC_shape_seqs)
        
def code_seqs(seqs):
    # materialize for now, until we fix the following to work with generator
    seqs = list(seqs)

    one_hot_coded_seqs = one_hot_encode_sequences(seqs)
    n_seqs = one_hot_coded_seqs.shape[0]
    seq_length = one_hot_coded_seqs.shape[2]
    shape_coded_fwd_seqs, shape_coded_RC_seqs = None, None
    #( shape_coded_fwd_seqs, shape_coded_RC_seqs 
    #  ) = code_seqs_shape_features(seqs, seq_length, n_seqs)
    return CodedSeqs(
        one_hot_coded_seqs, shape_coded_fwd_seqs, shape_coded_RC_seqs)

def estimate_chem_affinities_for_selex_experiment(
        background_seqs, num_rounds, binding_model, dna_conc, prot_conc):
    all_chem_affinities = []
    pool = FixedLengthDNASequences(background_seqs)
    for i in xrange(num_rounds):
        all_chem_affinities.append(
            est_chem_potential( pool, binding_model, dna_conc, prot_conc ) )
    return np.array(all_chem_affinities, dtype=theano.config.floatX)

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

def find_pwm_from_starting_alignment(seqs, counts, max_num_seqs=5000):
    assert counts.shape[0] == 4
    bs_len = counts.shape[1]
    
    # code the sequences
    all_binding_sites = []
    all_weights = []
    for seq in seqs[:max_num_seqs]:
        binding_sites = list(enumerate_binding_sites(seq, bs_len))
        all_binding_sites.append( binding_sites )
        weights = np.array([1.0/len(binding_sites)]*len(binding_sites))
        all_weights.append(weights)

    # iterate over the alignments
    prev_counts = counts.copy()
    for i in xrange(10):
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
    pyTFbindtools.log("Found initial model")

    return pwm

_SelexData = namedtuple('SelexData', ['bg_seqs', 'rnd_seqs'])
class SelexData(_SelexData):
    @property
    def last_rnd_index(self):
        return max(self.rnd_seqs.iterkeys())

    @property
    def first_rnd_index(self):
        return min(self.rnd_seqs.iterkeys())

    @property    
    def last_rnd(self):
        return self.rnd_seqs[self.last_rnd_index]

    @property    
    def first_rnd(self):
        return self.rnd_seqs[self.first_rnd_index]

class PartitionedAndCodedSeqs(object):
    @property
    def max_rnd(self):
        return max(self.coded_seqs.keys())

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
            n_partitions = min(17, max(
                5, sum(len(seqs) for seqs in rnds_and_seqs.itervalues())/10000))
        self.n_partitions = n_partitions
        if self.n_partitions <= 3: 
            raise ValueError, "Need at least 3 partitions (test, train, validation)"
        
        # store the full coded sequence arrays
        self.coded_seqs = {}
        for rnd, seqs in rnds_and_seqs.iteritems():
            self.coded_seqs[rnd] = code_seqs(seqs)
            assert self.seq_length == self.coded_seqs[rnd].seq_length
        self.coded_bg_seqs = code_seqs(background_seqs)
        assert self.seq_length == self.coded_bg_seqs.seq_length
        
        # store views to partitioned subsets of the data 
        self._partitioned_data = [{} for i in xrange(self.n_partitions)]
        for rnd, coded_seqs in self.coded_seqs.iteritems():
            for partition_index, partition in enumerate(
                    coded_seqs.iter_coded_seq_splits(self.n_partitions)):
                self._partitioned_data[partition_index][rnd] = partition

        # partition the background sequences.
        if self.use_full_background_for_part_fn:
            self._partitioned_bg_data = [
                self.coded_bg_seqs for i in xrange(self.n_partitions)]
        else:
            self._partitioned_bg_data = list(
                self.coded_bg_seqs.iter_coded_seq_splits(self.n_partitions))
        
        self.test = SelexData(
            self._partitioned_bg_data[0], self._partitioned_data[0])
        self.validation = SelexData(
            self._partitioned_bg_data[1], self._partitioned_data[1])
        self.train = [
            SelexData(self._partitioned_bg_data[i], self._partitioned_data[i])
            for i in xrange(2, self.n_partitions)
        ]

def estimate_dg_matrix_with_adadelta(
        partitioned_and_coded_rnds_and_seqs,
        init_ddg_array, init_ref_energy, init_chem_affinities,
        dna_conc, prot_conc,
        ftol=1e-12):    
    max_rnd = partitioned_and_coded_rnds_and_seqs.max_rnd

    def calc_penalty(ref_energy, ddg_array):
        penalty = 0
        
        # Penalize models with non-physical mean affinities
        new_mean_energy = ref_energy + ddg_array.sum()/3
        if CONSTRAIN_MEAN_ENERGY:
            penalty += (new_mean_energy - EXPECTED_MEAN_ENERGY)**2

        # Penalize non-physical differences in base affinities
        if CONSTRAIN_BASE_ENERGY_DIFF:
            energy_diff = ddg_array.max(1) - ddg_array.min(1)
            penalty += (energy_diff[
                (energy_diff > MAX_BASE_ENERGY_DIFF)]**2).sum()
        #return 0
        return penalty

    def extract_data_from_array(x):
        ref_energy = x[0]
        num_ddg_entries = init_ddg_array.shape[0]*init_ddg_array.shape[1]
        ddg_array = x[1:1+num_ddg_entries].reshape(
            init_ddg_array.shape).astype('float32').view(
            DeltaDeltaGArray)
        chem_affinities = x[1+num_ddg_entries:] #np.array([-11.0]*max_rnd, dtype='float32')
        return ref_energy, ddg_array, chem_affinities

    def pack_data_into_array(output, ref_energy, ddg_array, chem_affinities):
        output[0] = ref_energy
        output[1:1+len(ddg_array.ravel())] = ddg_array.ravel()
        output[1+len(ddg_array.ravel()):] = chem_affinities
        return output
    
    def f_dg(x, data):
        """Calculate the loss.
        
        """
        ref_energy, ddg_array, chem_affinities = extract_data_from_array(x)
        rv = log_lhd.calc_log_lhd(
            ref_energy, 
            ddg_array, 
            chem_affinities,
            data,
            dna_conc, 
            prot_conc)
        #penalty = calc_penalty(ref_energy, ddg_array)
        return -rv # + penalty

    def f_grad(x, data):
        """Calculate the loss.
        
        """
        ref_energy, ddg_array, chem_affinities = extract_data_from_array(x)
        ref_energy_grad, ddg_grad, chem_pot_grad = log_lhd.calc_grad(
            ref_energy, 
            ddg_array, 
            chem_affinities,
            data,
            dna_conc, 
            prot_conc)
        rv = np.array([ref_energy_grad.tolist(),]  +
                      ddg_grad.ravel().tolist() +
                      chem_pot_grad.tolist(), 
                      dtype='float32')
        return -rv

    def f_grad2(x, data):
        """Calculate the loss.
        
        """
        return approx_fprime(
            x0, f_dg, 1e-3, data).astype('float32')

    def update_x(x, delta_x):
        # update the reference energy
        x[0] += delta_x.clip(-1, 1)[0] #grad #delta
        # update teh base contributions
        x[1:-len(init_chem_affinities)] += delta_x.clip(
            -0.1, 0.1)[1:-len(init_chem_affinities)] 
        # update the chemical affinities
        x[-len(init_chem_affinities):] += delta_x.clip(
            -1, 1)[-len(init_chem_affinities):] #grad #delta
        ref_energy, ddg_array, chem_affinities = extract_data_from_array(x)
        #print ref_energy
        #print ddg_array
        ref_energy += ddg_array[:,:4].min(1).sum()
        ddg_array[:,:4] -= ddg_array[:,:4].min(1)[:,None]
        #print ref_energy
        #print ddg_array
        x = pack_data_into_array(x, ref_energy, ddg_array, chem_affinities)
        return x

    # ada delta
    validation_lhds = []
    train_lhds = []
    xs = []
    def ada_delta(x0):
        # from http://arxiv.org/pdf/1212.5701.pdf
        e = 1e-6
        p = 0.50
        grad_sq = np.zeros(len(x0), dtype='float32')
        delta_x_sq = np.ones(len(x0), dtype='float32')
        
        eps = 1.0
        num_small_decreases = 0
        valid_train_indices = range(
            len(partitioned_and_coded_rnds_and_seqs.train))
        random.shuffle(valid_train_indices)
        for i in xrange(MAX_NUM_ITER):
            if len(valid_train_indices) == 0:
                valid_train_indices = range(
                    len(partitioned_and_coded_rnds_and_seqs.train))
                random.shuffle(valid_train_indices)
            train_index = valid_train_indices.pop()
            assert train_index < len(partitioned_and_coded_rnds_and_seqs.train)
            grad = f_grad(
                x0.astype('float32'), 
                partitioned_and_coded_rnds_and_seqs.train[train_index])
            grad_sq = p*grad_sq + (1-p)*(grad**2)
            delta_x = -grad*np.sqrt(delta_x_sq + e)/np.sqrt(grad_sq + e)  # 
            delta_x_sq = p*delta_x_sq + (1-p)*(delta_x**2)
            x0 = update_x(x0, delta_x)
            train_lhd = -f_dg(
                x0, partitioned_and_coded_rnds_and_seqs.train[train_index])
            validation_lhd = -f_dg(
                x0, partitioned_and_coded_rnds_and_seqs.validation)

            ref_energy, ddg_array, chem_affinities = extract_data_from_array(x0)
            print
            print grad[0].round(2)
            print grad[1:-len(chem_affinities)
                ].reshape(ddg_array.shape).round(2)
            print grad[-len(chem_affinities):].round(2)
            print 
            print delta_x[0].round(2)
            print delta_x[1:-len(chem_affinities)
                ].reshape(ddg_array.shape).round(2)
            print delta_x[-len(chem_affinities):].round(2)

            #print f_grad2(
            #    x0, partitioned_and_coded_rnds_and_seqs.train[train_index])
            summary = ddg_array.summary_str(ref_energy)
            summary += "\n" + "\n".join((
                "Chem Affinities: %s" % (str(chem_affinities.round(2))),
                "Train: %s (%i)" % (train_lhd, train_index),
                "Validation: %s" % validation_lhd,
                "Grad L2 Norm: %.2f" % math.sqrt((grad**2).sum())
                ))

            pyTFbindtools.log(summary, 'DEBUG')
            
            train_lhds.append(train_lhd)
            validation_lhds.append(validation_lhd)
            xs.append(x0)
            min_iter = 4*len(partitioned_and_coded_rnds_and_seqs.train)
            best = float('-inf')
            if i > 2*min_iter:
                old_median = np.median(validation_lhds[-2*min_iter:-min_iter])
                new_max = max(validation_lhds[-min_iter:])
                if new_max > best: best = new_max
                print "Stop Crit:", old_median, new_max, new_max-old_median, best
                if old_median - new_max > -1e-2 or best - 1e-2 > new_max:
                    break

        x_hat_index = np.argmax(np.array(validation_lhds))
        return xs[x_hat_index]
    
    bs_len = init_ddg_array.motif_len    

    x0 = np.zeros(1 
                  + init_ddg_array.shape[0]*init_ddg_array.shape[1] 
                  + len(init_chem_affinities), 
                  dtype=theano.config.floatX)
    x0 = pack_data_into_array(
        x0, init_ref_energy, init_ddg_array, init_chem_affinities)

    x = ada_delta(x0)
    ref_energy, ddg_array, chem_affinities = extract_data_from_array(x)
    validation_lhd = log_lhd.calc_log_lhd(
        ref_energy, 
        ddg_array, 
        chem_affinities,
        partitioned_and_coded_rnds_and_seqs.validation,
        dna_conc, 
        prot_conc)

    return ddg_array, ref_energy, chem_affinities, validation_lhds, validation_lhd

def sample_random_seqs(n_sims, seq_len):
    return ["".join(random.choice('ACGT') for j in xrange(seq_len))
            for i in xrange(n_sims)]

def find_best_shift(coded_seqs, binding_model, coded_bg_seqs=None):
    def find_flanking_base_cnts(seqs, base_offset):
        # calculate the ddg energies for all binding sites. We use this to align
        # the binding sities within the sequences
        # find the index of the best offset
        energies = np.array(
            binding_model.score_seqs_binding_sites(seqs, 'MAX'))
        best_offsets = np.argmin(energies, 1)
        
        ## find the binding sites which align to the sequence boundary. We must 
        ## remove these because the flanking sequence is not random, and so it
        ## will bias the results
        # deal with left shifts
        if base_offset > 0:
            base_offset += (binding_model.motif_len - 1)
        base_indices = best_offsets + base_offset
        # find which offsets fit inside of the random portion of the sequence
        valid_indices = (base_indices > 0)&(base_indices < seq_len)
        base_cnts = seqs.one_hot_coded_seqs[
            valid_indices,base_indices[valid_indices],:].sum(0)
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

    seq_len = coded_seqs.seq_length
    if coded_bg_seqs == None:
        coded_bg_seqs = sample_random_coded_seqs(
            coded_seqs.shape[0], seq_len)

    ### Code to downsample - I dont think that we want this but the 
    ### entropy calculation isn't perfect so Ill leave it
    #sample_size = min(coded_seqs.shape[0], coded_bg_seqs.shape[0])
    #coded_seqs = coded_seqs[
    #    np.random.choice(coded_seqs.shape[0], sample_size),:,:]
    #coded_bg_seqs = coded_bg_seqs[
    #    np.random.choice(coded_bg_seqs.shape[0], sample_size),:,:]
    
    entropies_and_offsets = []
    for offset in chain(xrange(max(-3, -seq_len+binding_model.motif_len+1), 0), 
                        xrange(1,min(4, seq_len-binding_model.motif_len))):
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
    for _, _, shift, entropy, bg_entropy in entropies_and_offsets:
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
    'energetic_model', 
    'lhd_path', 'lhd_hat', 
    'prev_validation_lhd', 'new_validation_lhd']
)

def extend_binding_model(mo, seqs, bg_seqs):
    pyTFbindtools.log("Finding best shift", 'VERBOSE')
    shift_type = find_best_shift(seqs, mo, bg_seqs)
    if shift_type == None:
        return None

    if shift_type == 'LEFT':
        pyTFbindtools.log("Adding left base to motif", level='VERBOSE' )
        ddg_array = np.vstack(
            (np.zeros(mo.ddg_array.shape[1], dtype='float32'), 
             mo.ddg_array)
        ).view(DeltaDeltaGArray)
    elif shift_type == 'RIGHT':
        pyTFbindtools.log("Adding right base to motif", level='VERBOSE' )
        ddg_array = np.vstack(
            (mo.ddg_array, 
             np.zeros(mo.ddg_array.shape[1], dtype='float32'))
        ).view(DeltaDeltaGArray)
    else:
        assert False, "Unrecognized shift type '%s'" % shift_type
    return EnergeticDNABindingModel(
        mo.ref_energy, ddg_array, **mo.meta_data)

def progressively_fit_model(
        partitioned_and_coded_rnds_and_seqs,
        initial_model, 
        chem_affinities,
        dna_conc, 
        prot_conc):    
    pyTFbindtools.log("Compiling likleihood function", 'VERBOSE')
    log_lhd.calc_log_lhd, log_lhd.calc_grad = log_lhd.theano_log_lhd_factory(
        partitioned_and_coded_rnds_and_seqs.train[0])
    pyTFbindtools.log("Starting optimization", 'VERBOSE')

    curr_mo = initial_model
    while True:
        i = curr_mo.motif_len
        prev_validation_lhd = log_lhd.calc_log_lhd(
            curr_mo.ref_energy, curr_mo.ddg_array, chem_affinities,
            partitioned_and_coded_rnds_and_seqs.validation,
            dna_conc, prot_conc)
        pyTFbindtools.log("Starting lhd: %.2f" % prev_validation_lhd, 'VERBOSE')
        
        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, chem_affinities, lhd_path, lhd_hat 
            ) = estimate_dg_matrix_with_adadelta(
                partitioned_and_coded_rnds_and_seqs,
                curr_mo.ddg_array, curr_mo.ref_energy, chem_affinities,
                dna_conc, prot_conc)

        summary = ddg_array.summary_str(ref_energy)
        #summary += "\n" + "\n".join((
        #    "Train: %s (%i)" % (train_lhd, train_index),
        #    "Validation: %s" % validation_lhd,
        #    "Grad L2 Norm: %.2f" % math.sqrt((grad**2).sum())
        #    ))
        pyTFbindtools.log(summary, 'VERBOSE')

        new_validation_lhd = log_lhd.calc_log_lhd(
            ref_energy, 
            ddg_array, 
            chem_affinities,
            partitioned_and_coded_rnds_and_seqs.validation,
            dna_conc,
            prot_conc)

        pyTFbindtools.log("Prev: %.2f\tCurr: %.2f\tDiff: %.2f" % (
            prev_validation_lhd, 
            new_validation_lhd, 
            new_validation_lhd-prev_validation_lhd
        ), 'VERBOSE')

        curr_mo = EnergeticDNABindingModel(
            ref_energy, ddg_array, **initial_model.meta_data)

        yield FitSelexModel( curr_mo, 
                             lhd_path, 
                             lhd_hat, 
                             prev_validation_lhd, 
                             new_validation_lhd)

        if ( curr_mo.motif_len >= MAX_BS_LEN
             or curr_mo.motif_len+1 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break
        curr_mo = extend_binding_model(
            curr_mo, 
            partitioned_and_coded_rnds_and_seqs.validation.last_rnd,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs)

    return
