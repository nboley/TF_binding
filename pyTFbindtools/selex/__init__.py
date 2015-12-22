import os, sys
import math
import random

from itertools import izip, chain

from collections import defaultdict, namedtuple

import numpy as np
np.set_printoptions(precision=4)
np.random.seed(0)

from scipy.optimize import brentq, approx_fprime, bracket, minimize_scalar
from scipy.stats import ttest_ind

import theano

from pyDNAbinding.binding_model import (
    FixedLengthDNASequences, est_chem_potential_from_affinities, calc_occ,
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

MAX_BS_LEN = 12

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

        if self.shape_coded_fwd_seqs is not None:
            assert self.shape_coded_fwd_seqs is not None 
            self.seqs = np.dstack([
                self.one_hot_coded_seqs, 
                self.shape_coded_fwd_seqs, 
                self.shape_coded_RC_seqs
            ])
        else:
            self.seqs = self.one_hot_coded_seqs

    def iter_coded_seq_splits(self, n_partitions):
        for (one_hot_seqs, fwd_shape_seqs, RC_shape_seqs
            ) in izip(self._array_split_or_none(self.one_hot_coded_seqs, n_partitions),
                      self._array_split_or_none(self.shape_coded_fwd_seqs, n_partitions),
                      self._array_split_or_none(self.shape_coded_RC_seqs, n_partitions)
                ):
            yield CodedSeqs(one_hot_seqs, fwd_shape_seqs, RC_shape_seqs)
        
def code_seqs(seqs, include_shape):
    # materialize for now, until we fix the following to work with generator
    seqs = list(seqs)

    one_hot_coded_seqs = one_hot_encode_sequences(seqs)
    n_seqs = one_hot_coded_seqs.shape[0]
    seq_length = one_hot_coded_seqs.shape[1]
    if not include_shape:
        shape_coded_fwd_seqs, shape_coded_RC_seqs = None, None
    else:
        ( shape_coded_fwd_seqs, shape_coded_RC_seqs 
            ) = code_seqs_shape_features(seqs, seq_length, n_seqs)
    return CodedSeqs(
        one_hot_coded_seqs, shape_coded_fwd_seqs, shape_coded_RC_seqs)

def estimate_chem_affinities_for_selex_experiment(
        bg_seqs, num_rounds, binding_model, dna_conc, prot_conc):
    # if this is a numpy array, assume that they're one hot encoded seqs
    if isinstance(bg_seqs, CodedSeqs):
        affinities = -(np.array(
            binding_model.score_seqs_binding_sites(bg_seqs, 'MAX')).max(1))
    else:
        bg_seqs = FixedLengthDNASequences(bg_seqs)
        affinities = -(bg_seqs.score_binding_sites(binding_model, 'MAX').max(1))
    
    all_chem_affinities = []
    weights = np.ones(len(affinities), dtype=float)
    for i in xrange(num_rounds):
        weights /= weights.sum()
        chem_pot = est_chem_potential_from_affinities(
            affinities, dna_conc, prot_conc, weights )
        weights *= calc_occ(chem_pot, affinities)
        all_chem_affinities.append(chem_pot)
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

    @property
    def have_shape_features(self):
        return self.bg_seqs.shape_coded_fwd_seqs is not None

class PartitionedAndCodedSeqs(object):
    @property
    def max_rnd(self):
        return max(self.coded_seqs.keys())

    def __init__(self, 
                 rnds_and_seqs, 
                 background_seqs, 
                 include_shape_features,
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
            self.coded_seqs[rnd] = code_seqs(seqs, include_shape_features)
            assert self.seq_length == self.coded_seqs[rnd].seq_length
        self.coded_bg_seqs = code_seqs(background_seqs, include_shape_features)
        assert self.seq_length == self.coded_bg_seqs.seq_length
        self.data = SelexData(
            self.coded_bg_seqs, self.coded_seqs)
        
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

class AdamOptimizer():
    def __init__(self, x0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = np.zeros(x0.size, dtype='float32')
        self.v = np.zeros(x0.size, dtype='float32')
        self.t = 0

    def __call__(self, grad):
        self.t += 1

        self.m = self.beta1*self.m + (1-self.beta1)*grad
        #self.m /= (1-self.beta1**self.t)

        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        #self.v /= (1-self.beta2**self.t)

        return -(self.alpha/math.log(self.t+2))*self.m/(
            np.sqrt(self.v) + self.eps)

class AdaDeltaOptimizer():
    # from http://arxiv.org/pdf/1212.5701.pdf
    def __init__(self, x0, p=0.50, e=1e-6):
        self.p = p
        self.e = e

        self.grad_sq = np.zeros(x0.size, dtype='float32')
        self.delta_x_sq = np.zeros(x0.size, dtype='float32')
        self.t = 0

    def __call__(self, grad):
        self.t += 1
        self.grad_sq = self.p*self.grad_sq + (1-self.p)*(grad**2)
        delta_x = -grad*np.sqrt(
            self.delta_x_sq + self.e)/np.sqrt(grad_sq + e)
        self.delta_x_sq = self.p*self.delta_x_sq + (1-self.p)*(delta_x**2)
        return delta_x

class LineSearchOptimizer():
    def __init__(self, x0, e=1e-4, max_update=0.2):
        ref_energy, ddg_array, chem_affinities = x0.extract()
        self.e = e
        self._ddg_array_size = ddg_array.size

    def __call__(self, grad, ddg_hessian=None):
        if hessian is None:
            # initialize it to 0, because ones will be added in the inverse 
            ddg_hessian = 0
        inverse_hessian = np.linalg.inv(
            ddg_hessian + np.eye(ddg_array.size, dtype='float32'))
        inverse_hessian_times_grad = inverse_hessian.dot(ddg_grad.ravel())

        # normalize the update 
        if np.abs(inverse_hessian_times_grad).max() > self.max_update:
            inverse_hessian_times_grad = (
                self.max_update*inverse_hessian_times_grad.copy()
                /np.abs(inverse_hessian_times_grad).max()
            )

        def f(a): 
            rv = log_lhd.calc_log_lhd(
                ref_energy, 
                ddg_array-(a*inverse_hessian_times_grad).reshape(ddg_array.shape), 
                chem_affinities,
                data,
                dna_conc, 
                prot_conc,
                True)
            rv = rv if np.isfinite(rv) else float('-inf')
            #print a, rv
            return rv
            #return -rv

        ## Stupid descent algorithm - reduces over fitting from a proper 
        ## line search
        a = 1.0
        f_0 = f(0)
        f_curr = f(a)
        while f_curr < f_0 and a > e:
            a /= 2
            f_curr = f(a)
        rv = (np.zeros(1, dtype='float32'), 
              a*inverse_hessian_times_grad,
              np.zeros(chem_affinities.shape, dtype='float32').ravel())
        return np.concatenate(rv).ravel()

        ## Proper line search
        #rv = minimize_scalar(
        #    f, bounds=[e, 0.1], method='bounded', options={'ftol': 1.0})
        #print rv
        #step = np.zeros(ddg_grad.size) + e #(rv.x if np.isfinite(rv.fun) else e)
        #ddg_update = (
        #    e*inverse_hessian_times_grad
        #    #-(step*ddg_grad.ravel() + step.dot(ddg_hessian)*step)
        #).reshape(ddg_array.shape)


def hessian(x, jac, epsilon=1e-6):
    """Numerical approximation to the Hessian
    Parameters
    ------------
    x: array-like
        The evaluation point
    jac: 
        jacobian of the function
    epsilon: float
        The size of the step
    """
    N = x.size
    h = np.zeros((N,N))
    df_0 = jac(x)
    for i in xrange(N):
        xx0 = 1.*x[i]
        x[i] = xx0 + epsilon
        df_1 = jac(x)
        h[i,:] = (df_1 - df_0)/epsilon
        x[i] = xx0
    return h

class PackedModelParams(np.ndarray):
    """Store packed model parameters. 
    
    Most optimization routines require an unraveled array of model parameters. 
    This class stores such an array, and contains methods to unpack/pack the
    data into a natural representation.
    """
    def __new__(cls, ref_energy, ddg_array, chem_affinities):
        rv = np.zeros(
            1 + ddg_array.size + chem_affinities.size, 
            dtype='float32').view(cls)
        return rv
    
    def __init__(self, ref_energy, ddg_array, chem_affinities):
        self._ddg_array_size = ddg_array.size
        self._ddg_array_shape = ddg_array.shape
        self.update(ref_energy, ddg_array, chem_affinities)

    @property
    def ref_energy(self):
        return self[0]

    @property
    def ddg_array(self):
        return self[1:self._ddg_array_size+1].reshape(
            self._ddg_array_shape).view(DeltaDeltaGArray).copy()

    @property
    def chem_affinities(self):
        return np.array(self[1+self._ddg_array_size:]).copy()
    
    def extract(self):
        return self.ref_energy, self.ddg_array, self.chem_affinities

    def update(self, ref_energy, ddg_array, chem_affinities):
        self[0] = ref_energy
        self[1:1+self._ddg_array_size] = ddg_array.ravel()
        self[1+self._ddg_array_size:] = chem_affinities
        return self

    def copy(self):
        return PackedModelParams(
            self.ref_energy, self.ddg_array, self.chem_affinities)

def estimate_dg_matrix(
        partitioned_and_coded_rnds_and_seqs,
        init_ddg_array, init_ref_energy, init_chem_affinities,
        dna_conc, prot_conc,
        ftol=1e-12):    
    def f_dg(x, data, penalized=True):
        """Calculate the loss.
        
        """
        rv = log_lhd.calc_log_lhd(
            x.ref_energy, 
            x.ddg_array, 
            x.chem_affinities,
            data,
            dna_conc, 
            prot_conc,
            penalized)
        return rv

    def f_analytic_grad(x, data):
        """Calculate the loss.
        
        """
        ref_energy_grad, ddg_grad, chem_pot_grad = log_lhd.calc_grad(
            x.ref_energy, 
            x.ddg_array, 
            x.chem_affinities,
            data,
            dna_conc, 
            prot_conc)
        return PackedModelParams(ref_energy_grad, ddg_grad, chem_pot_grad)

    def f_FD_grad(x, data):
        """Calculate the loss.
        
        """
        return approx_fprime(
            x, f_dg, 1e-3, data).astype('float32')

    def f_grad(x, data):
        """Calculate the loss.
        
        """
        #return f_FD_grad(x, data)
        grad = f_analytic_grad(x, data)
        assert np.isfinite(grad).all()
        return grad

    def f_hessian(x, data):
        grad = f_grad(x, data)
        hessian = log_lhd.calc_hessian(
            x.ref_energy, 
            x.ddg_array, 
            x.chem_affinities,
            data,
            dna_conc, 
            prot_conc)
        # np.fill_diagonal(rv, grad**2, wrap=False)
        return PackedModelParams(
            0.0, hessian, np.zeros(chem_affinities.shape, dtype='float32'))
    
    def zero_energies(x, max_num_zeros=2, zeroing_base_thresh=2.0):
        ref_energy, ddg_array, chem_affinities = x.extract()
        for j in xrange(2):
            energy_diffs = (
                ddg_array.base_portion.max(1) - ddg_array.base_portion.min(1) )
            max_diff_index = energy_diffs.argmax()
            if energy_diffs[max_diff_index] < zeroing_base_thresh: 
                break
            ref_energy += ddg_array.base_portion[max_diff_index,:].mean()
            ddg_array.base_portion[max_diff_index,:] = 0
        x.update(ref_energy, ddg_array, chem_affinities)
        return x
    
    def update_x(x, delta_x, max_update=0.20):
        if np.abs(delta_x).max() > max_update:
            delta_x = max_update*delta_x.copy()/np.abs(delta_x).max()
        # zero the reference energy and chem affinity updates
        delta_x[0] = 0
        delta_x[-len(x.chem_affinities):] = 0
        # update x
        x -= delta_x

        # normalize the base contributions
        ref_energy, ddg_array, chem_affinities = x.extract()
        ref_energy = ref_energy + ddg_array.base_portion.min(1).sum()
        ddg_array.base_portion[:,:] -= ddg_array.base_portion.min(1)[:,None]
        
        # update the parameter values
        return x.update(ref_energy, ddg_array, chem_affinities)

    def estimate_chemical_affinities(x0):
        mo = EnergeticDNABindingModel(x0.ref_energy, x0.ddg_array)
        return estimate_chem_affinities_for_selex_experiment(
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs, 
            partitioned_and_coded_rnds_and_seqs.max_rnd, 
            mo, dna_conc, prot_conc)
    
    # ada delta
    validation_lhds = []
    train_lhds = []
    xs = []
    def fit(x0):
        avg_delta_x = np.zeros(len(x0), dtype='float32')

        old_validation_lhd = float('-inf')

        num_burn_in_iter = 500
        min_iter_for_convergence = 500 
        rounds_since_update = 0        
        best = float('-inf')
        train_index = None
        valid_train_indices = range(
            len(partitioned_and_coded_rnds_and_seqs.train))
        random.shuffle(valid_train_indices)
        adam_opt = AdamOptimizer(x0)
        ada_delta_opt = AdaDeltaOptimizer(x0)
        line_search_opt = LineSearchOptimizer(x0)
        for i in xrange(MAX_NUM_ITER):
            if True:
                if len(valid_train_indices) == 0:
                    valid_train_indices = range(
                        len(partitioned_and_coded_rnds_and_seqs.train))
                    random.shuffle(valid_train_indices)
                train_index = valid_train_indices.pop()
                assert train_index < len(partitioned_and_coded_rnds_and_seqs.train)
                data = partitioned_and_coded_rnds_and_seqs.train[train_index]
            else:
                data = partitioned_and_coded_rnds_and_seqs.data

            # use the hessian informed line search
            grad = f_grad(x0, data)
            if False:
                delta_x = line_search_opt(grad)
            # use Adam
            if True:
                delta_x = adam_opt(grad)
            # use Ada delta
            if False:
                delta_x = ada_delta_opt(grad)

            # update the parameter 
            x0 = update_x(x0, delta_x)
            # store a running average to decide when we can stop
            avg_delta_x = 0.95*avg_delta_x + 0.05*delta_x

            # calcualte the chemical affinity imbalance to decide if they need
            # to be updated
            chem_affinity_imbalance = log_lhd.calc_log_unbnd_frac(
                x0.ref_energy, 
                x0.ddg_array, 
                x0.chem_affinities,
                partitioned_and_coded_rnds_and_seqs.validation,
                dna_conc, 
                prot_conc)

            #validation_lhd = -f_dg(
            #    x0, partitioned_and_coded_rnds_and_seqs.validation)
            train_lhd = 0 if train_index is None else -f_dg(
                x0, partitioned_and_coded_rnds_and_seqs.train[train_index])

            unpenalized_validation_lhd = f_dg(x0, data, True)
            #train_lhd = unpenalized_validation_lhd
                                    
            summary = "\n\n"
            summary += "Delta x\n" + str( 
                avg_delta_x[1:-len(x0.chem_affinities)].reshape(
                    x0.ddg_array.shape).round(6)) + "\n"
            summary += x0.ddg_array.summary_str(x0.ref_energy)
            summary += "\n" + "\n".join((
                "Chem Affinities: %s" % (str(x0.chem_affinities.round(2))),
                "Imbalance: %s" % (str(chem_affinity_imbalance.round(2))),
                "Energy Diffs: %s" % (
                    x0.ddg_array.base_portion.max(1)
                    -x0.ddg_array.base_portion.min(1)),
                "Train: %s (%s)" % (train_lhd, train_index),
                #"Validation: %s" % validation_lhd,
                #"Grad L2 Norm: %.2f" % math.sqrt((grad**2).sum()),
                "Delta X L1 Norm: %.4f" % (np.abs(avg_delta_x)).sum(),
                "Real Lhd: %s" % unpenalized_validation_lhd,
                "%i Stop Crit: %.2f\t%.3f\t%i\t%i" % ( 
                    i,
                    unpenalized_validation_lhd, best, 
                    rounds_since_update, min_iter_for_convergence)
            ))
            pyTFbindtools.log(summary, 'DEBUG')
            
            if i%(num_burn_in_iter/5) == 0 and i <= num_burn_in_iter:
                x0 = zero_energies(x0)
                        
            train_lhds.append(train_lhd) #train_lhd)
            validation_lhds.append(unpenalized_validation_lhd)
            xs.append(x0.copy())
            
            if i > num_burn_in_iter + 10:
                if np.isfinite(unpenalized_validation_lhd) \
                   and unpenalized_validation_lhd > best: 
                    best = unpenalized_validation_lhd
                    rounds_since_update = 0
                if ( np.abs(avg_delta_x).sum() < 0.01 ):
                    rounds_since_update += 1
                if rounds_since_update > min_iter_for_convergence:
                    break

        x_hat_index = np.argmax(np.array(validation_lhds))
        return xs[x_hat_index]
    
    bs_len = init_ddg_array.motif_len    

    x0 = PackedModelParams(
        init_ref_energy, init_ddg_array, init_chem_affinities)
    x = fit(x0)
    ref_energy, ddg_array, chem_affinities = x.extract()
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
    #pyTFbindtools.log("Compiling OLD likelihood function", 'VERBOSE')
    #log_lhd.numerical_log_lhd = log_lhd.numerical_log_lhd_factory()

    pyTFbindtools.log("Compiling likelihood function", 'VERBOSE')
    (log_lhd.calc_log_lhd, log_lhd.calc_grad, log_lhd.calc_hessian, log_lhd.calc_log_unbnd_frac
     ) = log_lhd.theano_log_lhd_factory(
        partitioned_and_coded_rnds_and_seqs.train[0])

    pyTFbindtools.log("Starting optimization", 'VERBOSE')
    curr_mo = initial_model
    while True:
        i = curr_mo.motif_len
        prev_validation_lhd = log_lhd.calc_log_lhd(
            curr_mo.ref_energy, curr_mo.ddg_array, chem_affinities,
            partitioned_and_coded_rnds_and_seqs.data, # validation
            dna_conc, prot_conc)
        pyTFbindtools.log("Starting lhd: %.2f" % prev_validation_lhd, 'VERBOSE')
        
        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, chem_affinities, lhd_path, lhd_hat 
            ) = estimate_dg_matrix(
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
             or curr_mo.motif_len+2 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break
        # extend the mnodel twice to break palindrome symmetry
        curr_mo = extend_binding_model(
            curr_mo, 
            partitioned_and_coded_rnds_and_seqs.validation.last_rnd,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs)
        curr_mo = extend_binding_model(
            curr_mo, 
            partitioned_and_coded_rnds_and_seqs.validation.last_rnd,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs)

    return
