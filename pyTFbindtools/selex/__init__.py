import os, sys
import math

from itertools import product, izip, chain

from collections import defaultdict, namedtuple

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as TT

from scipy.optimize import (
    minimize, minimize_scalar, brentq, approx_fprime )
from numpy.fft import rfft, irfft

import random

import pyTFbindtools

from ..motif_tools import (
    load_motifs, logistic, R, T, DeltaDeltaGArray, Motif, load_motif_from_text)

# ignore theano warnings
import warnings
warnings.simplefilter("ignore")

PARTITION_FN_SAMPLE_SIZE = 10000

CMP_LHD_NUMERATOR_CALCS = False
RANDOM_POOL_SIZE = None
CONVERGENCE_MAX_LHD_CHANGE = None
MAX_NUM_ITER = 1000

EXPECTED_MEAN_ENERGY = -3.0
CONSTRAIN_MEAN_ENERGY = True

CONSTRAIN_BASE_ENERGY_DIFF = True
MAX_BASE_ENERGY_DIFF = 8.0

USE_SHAPE = False

# during optimization, how much to account for previous values
MOMENTUM = None

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
#for seq, values in shape_data.iteritems():
#    print seq, values
#assert False

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

def code_seqs(seqs, motif_len, n_seqs=None, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    if n_seqs == None: n_seqs = len(seqs)
    subseq0 = code_sequence(next(iter(seqs)), motif_len)
    # leave 3 rows for each sequence base, and 6 for the shape params
    len_per_base = 3
    if USE_SHAPE:
        len_per_base += 6
    coded_seqs = np.zeros((n_seqs, len(subseq0), motif_len*len_per_base), 
                          dtype=theano.config.floatX)
    for i, seq in enumerate(seqs):
        for j, param_values in enumerate(code_sequence(seq, motif_len)):
            coded_seqs[i, j, :] = param_values
    if ON_GPU:
        return theano.shared(coded_seqs)
    else:
        return coded_seqs

sym_cons_dg = TT.scalar('cons_dg')
sym_chem_pot = TT.scalar('chem_pot')
sym_ddg = TT.vector('ddg')

# calculate the sum of log occupancies for a particular round, given a 
# set of energies
calc_rnd_lhd_num = theano.function([sym_chem_pot, sym_cons_dg, sym_ddg], -(
    TT.log(1.0 + TT.exp(
        (-sym_chem_pot + sym_cons_dg + sym_ddg)/(R*T))).sum())
)

def calc_lhd_numerators(
        seq_ddgs, chem_affinities, ref_energy):
    # the occupancies of each rnd are a function of the chemical affinity of
    # the round in which there were sequenced and each previous round. We 
    # loop through each sequenced round, and calculate the numerator of the log 
    # lhd 
    numerators = []
    for sequencing_rnd, seq_ddgs in enumerate(seq_ddgs):
        chem_affinity = chem_affinities[0]
        #numerator = np.log(logistic(-(-chem_affinity+ref_energy+seq_ddgs)/(R*T))).sum()
        numerator = calc_rnd_lhd_num(chem_affinity, ref_energy, seq_ddgs)
        for rnd in xrange(1, sequencing_rnd+1):
            #numerator += np.log(
            #    logistic(-(-chem_affinities[rnd]+ref_energy+seq_ddgs)/(R*T))).sum()
            numerator += calc_rnd_lhd_num(
                chem_affinities[rnd], ref_energy, seq_ddgs)
        numerators.append(numerator)
    
    if CMP_LHD_NUMERATOR_CALCS:
        print "NUMERATORS", "="*10
        print numerators
        print chem_affinities
        print ref_energy
        print
    
    return numerators

def sample_random_seqs(n_sims, seq_len):
    return ["".join(random.choice('ACGT') for j in xrange(seq_len))
            for i in xrange(n_sims)]

random_seqs = sample_random_seqs(PARTITION_FN_SAMPLE_SIZE, 20)

def calc_log_lhd_factory(partitioned_and_coded_rnds_and_seqs, dna_conc, prot_conc):
    dna_conc = np.array(dna_conc).astype('float32')
    prot_conc = np.array(prot_conc).astype('float32')

    calc_energy_fns = []
    sym_e = TT.vector()
    for rnds_and_coded_seqs in partitioned_and_coded_rnds_and_seqs:
        calc_energy_fns.append([])
        for x in rnds_and_coded_seqs:
            #calc_energy_fns.append(
            #    theano.function([sym_e], theano.sandbox.cuda.basic_ops.gpu_from_host(
            #        x.dot(sym_e).min(1)) ) )
            calc_energy_fns[-1].append(
                theano.function([sym_e], x.dot(sym_e).min(1)) )

    random_coded_seqs = code_seqs(
        random_seqs, partitioned_and_coded_rnds_and_seqs.bs_len, ON_GPU=True)
    expected_cnts = (4**partitioned_and_coded_rnds_and_seqs.seq_length)/len(random_seqs)

    calc_bg_energies = theano.function(
        [sym_e], random_coded_seqs.dot(sym_e).min(1))
    calc_occ = theano.function([sym_chem_pot, sym_e], 
        1 / (1 + TT.exp((-sym_chem_pot+sym_e)/(R*T)))
    )
    
    sym_occ = TT.vector()
    calc_denom = theano.function([sym_occ], 
        (sym_occ*expected_cnts).sum()
    )

    sym_part_fn = TT.vector()
    sym_u = TT.scalar('u')
    calc_chem_pot_sum_term = theano.function([sym_part_fn, sym_e, sym_u], 
        (sym_part_fn*dna_conc/(1+TT.exp(sym_e-sym_u))).sum()
    )

    def est_chem_potential(
            partition_fn, energies, dna_conc, prot_conc ):
        """Estimate chemical affinity for round 1.

        [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
        exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
        """
        def f(u):
            sum_term = calc_chem_pot_sum_term(partition_fn, energies, u)
            #sum_terms = partition_fn*dna_conc/(1+np.exp(energies-u))
            return prot_conc - math.exp(u) - sum_term #.sum()
        min_u = -1000
        max_u = 100 + math.log(prot_conc)
        rv = brentq(f, min_u, max_u, xtol=1e-4)
        return rv

    def calc_lhd_denominators_and_chem_pots(
            ref_energy, ddg_array, seq_len, n_bind_sites):
        # now calculate the denominator (the normalizing factor for each round)
        # calculate the expected bin counts in each energy level for round 0
        energies = ref_energy + calc_bg_energies(ddg_array)
        curr_occupancies = np.ones(len(energies), dtype='float32')
        
        denominators = []
        chem_pots = []
        for rnd in xrange(len(partitioned_and_coded_rnds_and_seqs)):
            chem_affinity = est_chem_potential(
                energies, curr_occupancies/curr_occupancies.sum(),
                dna_conc, prot_conc )
            chem_pots.append(chem_affinity)
            curr_occupancies *= calc_occ(chem_affinity, energies)
            denominators.append(np.log(calc_denom(curr_occupancies)))
        return chem_pots, denominators
    
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     partition_index):
        ref_energy = np.array(ref_energy).astype('float32')
        # score all of the sequences
        rnds_and_seq_ddgs = []
        for rnd, calc_energy in enumerate(calc_energy_fns[partition_index]):
            rnds_and_seq_ddgs.append( calc_energy(ddg_array) )

        # calcualte the denominators
        chem_affinities, denominators = calc_lhd_denominators_and_chem_pots(
            ref_energy, ddg_array,  
            partitioned_and_coded_rnds_and_seqs.seq_length,
            partitioned_and_coded_rnds_and_seqs.n_bind_sites)

        # calculate the numerators
        numerators = calc_lhd_numerators(
            rnds_and_seq_ddgs, chem_affinities, ref_energy)


        lhd = 0.0
        for rnd_num, rnd_denom, rnd_seq_ddgs in izip(
                numerators, denominators, rnds_and_seq_ddgs):
            lhd += rnd_num - len(rnd_seq_ddgs)*rnd_denom
        
        return lhd
    
    return calc_log_lhd        

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
    consensus = find_consensus_bind_site(rnds_and_seqs[-1], bs_len)
    pyTFbindtools.log("Found consensus %imer '%s'" % (bs_len, consensus))
    counts = np.zeros((4, bs_len))
    for pos, base in enumerate(consensus): 
        counts[base_map(base), pos] += 1000

    # find a pwm using the initial sixmer alignment
    pwm = find_pwm_from_starting_alignment(rnds_and_seqs[0], counts)
    return pwm

def calc_occ(seq_ddgs, ref_energy, chem_affinity):
    return logistic(-(-chem_affinity+ref_energy+seq_ddgs)/(R*T))

class PartitionedAndCodedSeqs(list):
    @staticmethod
    def partition_data(seqs):
        assert len(seqs) > 150
        n_partitions = max(5, len(seqs)/10000)
        partitioned_seqs = [[] for i in xrange(n_partitions)]
        for i, seq in enumerate(seqs):
            partitioned_seqs[i%n_partitions].append(seq)
        return partitioned_seqs

    def __init__(self, rnds_and_seqs, bs_len):
        self.seq_length = len(rnds_and_seqs[0][0])
        self.bs_len = bs_len
        
        self.extend(zip(*[
            [ code_seqs(rnd_seqs, bs_len)
              for rnd_seqs in self.partition_data(seqs)]
            for seqs in rnds_and_seqs]))
        self.validation = self[0]
        self.test = self[1]
        self.training = self[2:]

        self.n_bind_sites = self.validation[0].get_value().shape[1]

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
        ddg_array = x[1:].astype('float32').view(DeltaDeltaGArray)
        return ref_energy, ddg_array
    
    def f_dg(x, train_index):
        ref_energy, ddg_array = extract_data_from_array(x)
        rv = calc_log_lhd(ref_energy, ddg_array, train_index)
        penalty = calc_penalty(ref_energy, ddg_array)
        return -rv + penalty

    # ada delta
    test_lhds = []
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
            train_index = random.randint(
                2, len(partitioned_and_coded_rnds_and_seqs)-1)
            grad = approx_fprime(x0, f_dg, 1e-3, train_index)
            grad_sq = p*grad_sq + (1-p)*(grad**2)
            delta_x = -np.sqrt(delta_x_sq + e)/np.sqrt(
                grad_sq + e)*grad
            delta_x_sq = p*delta_x_sq + (1-p)*(delta_x**2)
            x0 += delta_x.clip(-2, 2) #grad #delta
            train_lhd = -f_dg(x0, train_index)
            test_lhd = -f_dg(x0, 1)
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
            debug_output.append("Test: %s" % test_lhd)
            debug_output.append(str(math.sqrt((grad**2).sum())))
            pyTFbindtools.log("\n".join(debug_output), 'DEBUG')
            
            train_lhds.append(train_lhd)
            test_lhds.append(test_lhd)
            xs.append(x0)
            min_num_iter = 10
            if i > 2*min_num_iter and (
                    sum(test_lhds[-2*min_num_iter:-min_num_iter])/min_num_iter
                    > sum(test_lhds[-min_num_iter:])/min_num_iter ):
                break

        x_hat_index = np.argmax(np.array(test_lhds))
        return xs[x_hat_index]
    
    bs_len = init_ddg_array.motif_len    
    calc_log_lhd = calc_log_lhd_factory(
        partitioned_and_coded_rnds_and_seqs, dna_conc, prot_conc)

    x0 = init_ddg_array.copy().astype('float32')
    if USE_SHAPE:
        x0 = np.append(x0, np.zeros(6*(len(x0)/3)))
    x0 = np.insert(x0, 0, init_ref_energy)
    x = ada_delta(x0)

    ref_energy, ddg_array = extract_data_from_array(x)
    test_lhd = calc_log_lhd(ref_energy, ddg_array, 1)

    return ddg_array, ref_energy, chem_pots, test_lhds, test_lhd
