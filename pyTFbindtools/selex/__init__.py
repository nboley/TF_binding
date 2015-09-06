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

CMP_LHD_NUMERATOR_CALCS = False
RANDOM_POOL_SIZE = None
CONVERGENCE_MAX_LHD_CHANGE = None
MAX_NUM_ITER = None

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

def est_partition_fn_fft(ref_energy, ddg_array, n_bind_sites, seq_len, n_bins=2**12):
    # make sure the number of bins is a power of two. This is two aboid extra
    # padding during the fft convolution
    if (n_bins & (n_bins-1)):
        raise ValueError, "The number of bins must be a power of two"

    # reset the motif data so that the minimum value in each column is 0
    min_energy = ddg_array.calc_min_energy(ref_energy)
    max_energy = ddg_array.calc_max_energy(ref_energy)
    step_size = (max_energy-min_energy+1e-6)/(n_bins-ddg_array.motif_len)
    
    # build the polynomial for each base
    fft_product = np.zeros(n_bins, dtype='float32')
    new_poly = np.zeros(n_bins, dtype='float32')
    # for each base, add to the polynomial
    for base_i, base_energies in enumerate(
            ddg_array.calc_base_contributions()):
        # add 1e-12 to avoid rounding errors
        nonzero_bins = np.array(
            ((base_energies-base_energies.min()+1e-6)/step_size).round(), 
            dtype=int)
        new_poly[nonzero_bins] = 0.25
        freq_poly = rfft(new_poly, n_bins)
        new_poly[nonzero_bins] = 0
        
        if base_i == 0:
            fft_product = freq_poly
        else:
            fft_product *= freq_poly
    
    # move back into polynomial coefficient space
    part_fn = irfft(fft_product, n_bins)
    ## Account for the energy being the minimum over multiple binding sites
    # insert a leading zero for the cumsum
    min_cdf = 1 - (1 - part_fn.cumsum())**n_bind_sites
    min_cdf = np.insert(min_cdf, 0, 0.0)
    min_pdf = np.array(np.diff(min_cdf), dtype='float32')
    # build the energy grid
    x = np.linspace(min_energy, min_energy+step_size*n_bins, n_bins);
    assert len(x) == n_bins
    return x, min_pdf


cached_coded_seqs = {}
def est_partition_fn_brute(ref_energy, ddg_array, n_bind_sites, seq_len):
    assert ddg_array.motif_len <= 8
    if ddg_array.motif_len not in cached_coded_seqs:
        cached_coded_seqs[ddg_array.motif_len] = code_seqs(
            product('ACGT', repeat=ddg_array.motif_len), 
            ddg_array.motif_len, 
            n_seqs=4**ddg_array.motif_len, 
            ON_GPU=False)
    coded_seqs = cached_coded_seqs[ddg_array.motif_len]
    
    energies = ref_energy + coded_seqs.dot(ddg_array).min(1)
    energies.sort()
    part_fn = np.ones(len(energies), dtype=float)/len(energies)
    min_cdf = 1 - (1 - part_fn.cumsum())**n_bind_sites
    #min_cdf = np.insert(min_cdf, 0, 0.0)
    min_pdf = np.array(np.diff(min_cdf), dtype='float32')
    return energies, min_cdf

def est_partition_fn_sampling(ref_energy, ddg_array, n_bind_sites, seq_len):
    n_sims = 10000
    key = ('SIM', ddg_array.motif_len)
    if key not in cached_coded_seqs:
        current_pool = ["".join(random.choice('ACGT') for j in xrange(seq_len))
                        for i in xrange(n_sims)]
        coded_seqs = code_seqs(current_pool, ddg_array.motif_len, ON_GPU=False)
        cached_coded_seqs[key] = coded_seqs
    coded_seqs = cached_coded_seqs[key]
    energies = ref_energy + coded_seqs.dot(ddg_array).min(1)
    energies.sort()
    part_fn = np.ones(len(energies), dtype=float)/len(energies)
    return energies, part_fn

#est_partition_fn = est_partition_fn_fft
#est_partition_fn = est_partition_fn_brute
est_partition_fn = est_partition_fn_sampling

def calc_occ(seq_ddgs, ref_energy, chem_affinity):
    return logistic(-(-chem_affinity+ref_energy+seq_ddgs)/(R*T))

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

def calc_lhd_denominators(
        ref_energy, ddg_array, chem_affinities, seq_len, n_bind_sites):
    # now calculate the denominator (the normalizing factor for each round)
    # calculate the expected bin counts in each energy level for round 0
    energies, partition_fn = est_partition_fn(
        ref_energy, ddg_array, seq_len, n_bind_sites)
    expected_cnts = (4**seq_len)*partition_fn 
    curr_occupancies = np.ones(len(energies), dtype='float32')
    denominators = []
    for rnd, chem_affinity in enumerate(chem_affinities):
        curr_occupancies *= logistic(-(-chem_affinity+energies)/(R*T))
        denominators.append( np.log((expected_cnts*curr_occupancies).sum()) )
    #print denominators
    return denominators

def calc_log_lhd_factory(partitioned_and_coded_rnds_and_seqs):    
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
    
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     rnds_and_chem_affinities,
                     partition_index):
        assert len(rnds_and_coded_seqs) == len(rnds_and_chem_affinities)
        ref_energy = np.array(ref_energy).astype('float32')
        rnds_and_chem_affinities = rnds_and_chem_affinities.astype('float32')
        # score all of the sequences
        rnds_and_seq_ddgs = []
        for rnd, calc_energy in enumerate(calc_energy_fns[partition_index]):
            rnds_and_seq_ddgs.append( calc_energy(ddg_array) )
        # calculate the numerators
        numerators = calc_lhd_numerators(
            rnds_and_seq_ddgs, rnds_and_chem_affinities, ref_energy)

        # calcualte the denominators
        denominators = calc_lhd_denominators(
            ref_energy, ddg_array, rnds_and_chem_affinities, 
            partitioned_and_coded_rnds_and_seqs.seq_length,
            partitioned_and_coded_rnds_and_seqs.n_bind_sites)

        lhd = 0.0
        for rnd_num, rnd_denom, rnd_seq_ddgs in izip(
                numerators, denominators, rnds_and_seq_ddgs):
            lhd += rnd_num - len(rnd_seq_ddgs)*rnd_denom

        return lhd
    
    return calc_log_lhd        

def init_param_space_lhs(samples, N=1000 ):
    """
    Initializes a starting location with Latin Hypercube Sampling

    ## Find random starting location
    x0 = ddg_array
    if CONSIDER_RANDOM_START:
        max_lhd = -1e100
        curr_x = None
        for i, x in enumerate(init_param_space_lhs(len(ddg_array))):
            lhd = -f(x)
            print i
            if lhd > max_lhd:
                max_lhd = lhd
                curr_x = x
        print "="*60
        f(x)
        print "="*60
        f(x0)
        
        if f(x) < f(x0):
            x0 = x
    """
    rng = np.random.mtrand._rand

    # Generate the intervals
    segsize = 1.0 / samples

    # Fill points uniformly in each interval
    rdrange = rng.rand(samples, N) * segsize
    rdrange += np.atleast_2d(
        np.linspace(0., 1., samples, endpoint=False)).T

    # Make the random pairings
    population = np.zeros_like(rdrange)

    for j in range(N):
        order = rng.permutation(range(samples))
        population[:, j] = rdrange[order, j]
    return (population.T - 0.5)*10

def estimate_dg_matrix(rnds_and_seqs, init_ddg_array, init_ref_energy,
                       dna_conc, prot_conc,
                       ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    
    def calc_penalty(ref_energy, ddg_array, chem_pots):
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

    def f_single_base_dg(x, base_pos, ref_energy):
        ddg_array[3*base_pos:3*(base_pos+1)] += x
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, 
            len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, ddg_array, chem_pots)
        penalty = calc_penalty(ref_energy, ddg_array, chem_pots)        
        ddg_array[3*base_pos:3*(base_pos+1)] -= x
        return -rv + penalty

    def f_dg(x):
        ref_energy = x[0]
        ddg_array = x[1:].astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, 
            len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, ddg_array, chem_pots)
        penalty = calc_penalty(ref_energy, ddg_array, chem_pots)
        
        """
        print ddg_array.consensus_seq()
        print chem_pots
        print "Ref:", ref_energy
        print "Mean:", ref_energy + x.sum()/3
        print "Min:", ddg_array.calc_min_energy(ref_energy)
        print ddg_array.calc_base_contributions().round(2)
        print rv
        print penalty
        print rv + penalty
        """
        return -rv + penalty

    def f_ref_energy(x):
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy+x, dna_conc, prot_conc,
            n_bind_sites, 
            len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy+x, ddg_array, chem_pots)
        penalty = calc_penalty(ref_energy+x, ddg_array, chem_pots)

        return -rv + penalty

    # ada delta
    def ada_delta(x0):
        # from http://arxiv.org/pdf/1212.5701.pdf
        e = 1e-6
        p = 0.99
        grad_sq = np.zeros(len(x0))
        delta_x_sq = np.zeros(len(x0))
        
        eps = 1.0
        num_small_decreases = 0
        for i in xrange(20):
            grad = approx_fprime(x0, f_dg, epsilon=1e-2)
            #grad /= np.abs(grad).sum()
            grad_sq = p*grad_sq + (1-p)*(grad**2)
            delta_x = -np.sqrt(delta_x_sq + e)/np.sqrt(
                grad_sq + e)*grad
            delta_x_sq = p*delta_x_sq + (1-p)*(delta_x**2)
            x0 += delta_x #.clip(-2, 2) #grad #delta
            print f_dg(x0)
            print math.sqrt((grad**2).sum())
            print grad
            print delta_x
            print x0[1:].view(DeltaDeltaGArray).calc_base_contributions().round(2)
            #raw_input()

        return

    x0 = init_ddg_array.copy().astype('float32')
    x0 = np.insert(x0, 0, init_ref_energy)
    ada_delta(x0)
    
    """
    # gradient descent
    eps = 1.0
    num_small_decreases = 0
    while num_small_decreases < 10:
        grad = approx_fprime(x0, f_dg, epsilon=eps)
        grad /= np.abs(grad).sum()
        def f(alpha):
            return f_dg(alpha*grad + x0)
        prev_fun = f(0)
        res = minimize_scalar(f, bounds=[-10,10])
        if abs(res.x) < eps: 
            eps /= 2
        elif abs(res.x) > 2*eps:
            eps *= 2
        if prev_fun - res.fun < 1e-6:
            num_small_decreases += 1
        else:
            num_small_decreases = 0
        print num_small_decreases, eps, res.fun, res.fun - prev_fun
        
        x0 += res.x*grad

    ddg_array = res.x[1:].astype('float32').view(DeltaDeltaGArray)    
    ref_energy = res.x[0]
    """
    
    # Nelder Meadx
    """
    res = minimize(f_dg, x0, tol=ftol, method='Nelder-Mead', # COBYLA  
                   options={'disp': False, 
                            'maxiter': 1, 
                            'xtol': 1e-3, 'ftol': 1e-2} )
    ddg_array = res.x[1:].astype('float32').view(DeltaDeltaGArray)    
    ref_energy = res.x[0]
    """
    
    ddg_array = init_ddg_array.copy()
    print "Initial DDG Array Shape:", ddg_array.shape
    ref_energy = np.array(init_ref_energy)
    iteration_number = 0
    tol = 1e-2
    #assert False
    # initialize the lhd changes to a large number so each base is updated
    # once in the first round. We add an additional weight for the ref energy
    prev_lhd = -f_ref_energy(0.0)
    lhd_changes = 1000*np.ones(ddg_array.motif_len+1)
    weights = lhd_changes/lhd_changes.sum()
    # until the minimum update tolerance is below the stop iteration threshold  
    while tol > 100*CONVERGENCE_MAX_LHD_CHANGE and iteration_number < MAX_NUM_ITER:
        iteration_number +=1 
        # find the base position to update next. We want to focus on the bases
        # that are givng the largest updates, but we also want the process to be
        # somewhat to avoid shifting the motif in a particular direction, so we 
        # choose the update base proportional to the update weights in the 
        # previous round
        print "Weights:", weights.round(2)
        base_index = np.random.choice(
            np.arange(ddg_array.motif_len+1),
            size=1,
            p=weights )
        
        #base_index = 6
        print "="*50
        print "Minimizing pos %i at tolerance %.e" % (base_index+1, tol)
        print "Prev lhd Changes:", '  '.join("%.2e" % x for x in lhd_changes)

        # if this is the ref energy position
        if base_index == ddg_array.motif_len:
            res = minimize_scalar(f_ref_energy,
                                  bounds=[-5,5], tol=tol )  
            new_lhd = -res.fun
            if new_lhd > prev_lhd:
                ref_energy += res.x
            else:
                print "STEP DIDNT IMPROVE THE LHD"
                new_lhd = prev_lhd
        # otherwise this is a base energy
        else:
            res = minimize(
                f_single_base_dg, np.zeros(3, dtype=float),
                args=[base_index, ref_energy], 
                method='COBYLA', tol=tol)
            new_lhd = -res.fun
            print new_lhd, prev_lhd
            if new_lhd > prev_lhd:
                ddg_array[3*base_index:3*(base_index+1)] += res.x
            else:
                print "STEP DIDNT IMPROVE THE LHD"
                new_lhd = prev_lhd

        ## print debugging information
        print "Consensus:", ddg_array.consensus_seq()        
        print "Change:", res.x
        print "Ref:", ref_energy
        print "Chem Pots:", est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        print "Mean:", ref_energy + ddg_array.sum()/3
        print "Min:", ddg_array.calc_min_energy(ref_energy)
        print ddg_array.calc_base_contributions().round(2)
        print "New Lhd", new_lhd
        print "Lhd Change", new_lhd - prev_lhd

        ## update the base selection weights
        new_lhd_change = (
            lhd_changes[base_index]*MOMENTUM + (1-MOMENTUM)*(new_lhd-prev_lhd) )
        # avoid a divide by zero
        lhd_changes += tol/10
        # don't update the base we just updated
        lhd_changes[base_index] = 0.0
        weights = lhd_changes/lhd_changes.sum()
        lhd_changes -= tol/10
        lhd_changes[base_index] = new_lhd_change
        # if we had a successful update, then make sure we try every other entry
        if new_lhd - prev_lhd > tol/100:
            lhd_changes += tol
        prev_lhd = new_lhd
        
        # if the change has been small enough, update the search tolerance
        if lhd_changes.max() < tol: 
            # increase the lhd changes to make sure that every base is explored 
            # during before reducing the tolerance again 
            lhd_changes += tol
            tol /= 100

    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, dna_conc, prot_conc,
        n_bind_sites, len(rnds_and_seqs))
    new_lhd = calc_log_lhd(ref_energy, ddg_array, chem_pots)

    return ddg_array, ref_energy, new_lhd

def estimate_chem_pots_w_lhd(rnds_and_seqs, ddg_array, 
                             dna_conc, prot_conc,
                             ref_energy, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    chem_pots_0 = est_chem_potentials(
        ddg_array, ref_energy, dna_conc, prot_conc,
        n_bind_sites, len(rnds_and_seqs))
    
    def f(x):
        chem_pots = est_chem_potentials(
            ddg_array, x[0], dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(x[0], ddg_array, chem_pots_0)
        """
        print x
        print ref_energy
        print rv
        print calc_log_lhd(ref_energy, ddg_array, chem_pots_0)
        print
        """
        return -rv + abs(x[0])
    """
    def f(x):
        rv = calc_log_lhd(ref_energy, ddg_array, x)
        print est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        print x
        print rv
        print calc_log_lhd(ref_energy, ddg_array, x)
        print
        return -rv
    """
    x0 = ref_energy #chem_pots_0
    res = minimize(f, x0, tol=ftol, method='Powell', # COBYLA  
                   options={'disp': False, 'maxiter': 50000} )
    return res.x, -f([res.x,])

def est_chem_potential(
        energy_grid, partition_fn, 
        dna_conc, prot_conc ):
    """Estimate chemical affinity for round 1.
    
    [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
    exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
    """
    def f(u):
        sum_terms = dna_conc*partition_fn/(1+np.exp(energy_grid-u))
        return prot_conc - math.exp(u) - sum_terms.sum()
    min_u = -1000
    max_u = 100 + math.log(prot_conc)
    rv = brentq(f, min_u, max_u, xtol=1e-4)
    return rv

def est_chem_potentials(ddg_array, ref_energy, dna_conc, prot_conc,
                        n_bind_sites, seq_len, num_rnds):
    energy_grid, partition_fn = est_partition_fn(
        ref_energy, ddg_array, n_bind_sites, seq_len)
    chem_pots = []
    for rnd in xrange(num_rnds):
        chem_pot = est_chem_potential(
            energy_grid, partition_fn,
            dna_conc, prot_conc )
        chem_pots.append(chem_pot)
        partition_fn *= logistic(-(-chem_pot+energy_grid)/(R*T))
        partition_fn = partition_fn/partition_fn.sum()
    return np.array(chem_pots, dtype='float32')

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

def build_random_read_energies_pool(pool_size, read_len, ddg_array, ref_energy, 
                                    store_seqs=False):
    seqs = []
    energies = np.zeros(pool_size, dtype='float32')
    for i in xrange(pool_size):
        if i%10000 == 0: 
            pyTFbindtools.log("Bootstrapped %i reads." % i, level='VERBOSE')
        seq = "".join(random.choice('ACGT') for j in xrange(read_len))
        # np.random.randint(4, size=read_len)
        if store_seqs: seqs.append(seq)
        coded_seq = code_sequence(seq, ddg_array.motif_len)
        energy = 1e100
        for subseq in coded_seq:
            energy = min(energy, ddg_array.dot(subseq))
        energies[i] = energy
    return energies, seqs

def bootstrap_lhd_numerators(initial_energy_pool, sim_sizes, 
                             chem_pots, ref_energy, ddg_array):
    curr_energy_pool = initial_energy_pool

    numerators = []
    if CMP_LHD_NUMERATOR_CALCS: 
        rnd_seq_ddgs = []
    for rnd, (sim_size, chem_pot) in enumerate(izip(sim_sizes, chem_pots)):
        ### update the bound sequence pool
        # calculate the occupancies 
        occs = calc_occ(curr_energy_pool, ref_energy, chem_pot)
        if isinstance(occs, float):
            occs = np.array([occs,])
        # cast to a double, and re-normalize to avoid errors in np.random.choice
        ps = np.array(occs/occs.sum(), dtype='double')
        ps = ps/ps.sum()
        seq_indices = np.random.choice(
            len(curr_energy_pool), size=sim_size,
            p=ps, replace=True)
        selected_energies = curr_energy_pool[seq_indices]
        if CMP_LHD_NUMERATOR_CALCS: 
            rnd_seq_ddgs.append(selected_energies)
        # calculate the addition to the lhd
        numerator = 0
        for inner_rnd in xrange(rnd+1):
            numerator += calc_rnd_lhd_num(
                chem_pots[inner_rnd].astype('float32'), 
                ref_energy.astype('float32'), 
                selected_energies)
        numerators.append(numerator)

        seq_indices = np.random.choice(
            len(curr_energy_pool), size=len(curr_energy_pool),
            p=ps, replace=True)
        curr_energy_pool = curr_energy_pool[seq_indices]


    if CMP_LHD_NUMERATOR_CALCS:
        print "NUMERATORS", "+"*10, numerators
        print chem_pots
        print ref_energy
        print ddg_array.calc_min_energy(ref_energy)
        print "ENERGIES"
        for rnd_ddgs in rnd_seq_ddgs:
            print sum(rnd_ddgs)/len(rnd_ddgs)
        numerators = calc_lhd_numerators(
            rnd_seq_ddgs, chem_pots.astype('float32'), ref_energy) 
        print "NUMERATORS", "="*10, numerators
    
    return sum(numerators)

def bootstrap_lhds(read_len,
                   ddg_array, ref_energy,
                   chem_pots,
                   sim_sizes,
                   pool_size=10000 ):
    bs_len = ddg_array.motif_len
    # multiply vby two for the reverse complements
    n_bind_sites = 2*(read_len-bs_len+1)

    # calculate the normalizing constant. This is only a function of the 
    # energy model and chemical affinities, so we only need to do this once
    # for all of the simulated samples
    denominators = calc_lhd_denominators(
        ref_energy, ddg_array, chem_pots, n_bind_sites)
    normalizing_constant = sum(cnt*denom for cnt, denom 
                               in izip(sim_sizes, denominators))

    # simualte the numerators. First we build random sequences, and then 
    # calculate their energies. This pool will remain the same throughout
    # the simulations (to avoid additional variance from the initial pool
    # sampling - the real pool is much larger than we can simualte so we
    # don't want the additional variance from re-sampling every bootstrap )
    initial_energy_pool, seqs = build_random_read_energies_pool(
        pool_size, read_len, ddg_array, ref_energy)
    lhds = []
    for i in xrange(10):
        numerator = bootstrap_lhd_numerators(
            initial_energy_pool, sim_sizes, 
            chem_pots, ref_energy, ddg_array)
        lhds.append( numerator - normalizing_constant )
    return lhds

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
        self.extend(zip(*[
            [ code_seqs(rnd_seqs, bs_len)
              for rnd_seqs in self.partition_data(seqs)]
            for seqs in rnds_and_seqs]))
        self.n_bind_sites = self[0][0].get_value().shape[1]

def estimate_dg_matrix_with_adadelta(
        partitioned_and_coded_rnds_and_seqs,
        init_ddg_array, init_ref_energy,
        dna_conc, prot_conc,
        ftol=1e-12):    
    def calc_penalty(ref_energy, ddg_array, chem_pots):
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
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            partitioned_and_coded_rnds_and_seqs.n_bind_sites, 
            partitioned_and_coded_rnds_and_seqs.seq_length, 
            len(partitioned_and_coded_rnds_and_seqs[0]))
        return ref_energy, chem_pots, ddg_array
    
    def f_dg(x, train_index):
        ref_energy, chem_pots, ddg_array = extract_data_from_array(x)
        rv = calc_log_lhd(ref_energy, ddg_array, chem_pots, train_index)
        penalty = calc_penalty(ref_energy, ddg_array, chem_pots)
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
                1, len(partitioned_and_coded_rnds_and_seqs)-1)
            grad = approx_fprime(x0, f_dg, 1e-3, train_index)
            grad_sq = p*grad_sq + (1-p)*(grad**2)
            delta_x = -np.sqrt(delta_x_sq + e)/np.sqrt(
                grad_sq + e)*grad
            delta_x_sq = p*delta_x_sq + (1-p)*(delta_x**2)
            x0 += delta_x.clip(-2, 2) #grad #delta
            train_lhd = -f_dg(x0, train_index)
            test_lhd = -f_dg(x0, 0)
            ref_energy, chem_pots, ddg_array = extract_data_from_array(x0)
            
            print ddg_array.consensus_seq()
            print chem_pots
            print "Ref:", ref_energy
            print "Mean:", ref_energy + ddg_array.sum()/3
            print "Min:", ddg_array.calc_min_energy(ref_energy)
            print ddg_array.calc_base_contributions().round(2)
            print "Train: ", train_lhd, "(%i)" % train_index
            print "Test:", test_lhd
            print math.sqrt((grad**2).sum())

            train_lhds.append(train_lhd)
            test_lhds.append(test_lhd)
            xs.append(x0)
            min_num_iter = 10
            if i > 2*min_num_iter:
                print sum(test_lhds[-2*min_num_iter:-min_num_iter])/min_num_iter, \
                    sum(test_lhds[-min_num_iter:])/min_num_iter
            if i > 2*min_num_iter and (
                    sum(test_lhds[-2*min_num_iter:-min_num_iter])/min_num_iter
                    > sum(test_lhds[-min_num_iter:])/min_num_iter ):
                break

        x_hat_index = np.argmax(np.array(test_lhds))
        return xs[x_hat_index]
    
    bs_len = init_ddg_array.motif_len    
    calc_log_lhd = calc_log_lhd_factory(partitioned_and_coded_rnds_and_seqs)

    x0 = init_ddg_array.copy().astype('float32')
    if USE_SHAPE:
        x0 = np.append(x0, np.zeros(6*(len(x0)/3)))
    x0 = np.insert(x0, 0, init_ref_energy)
    x = ada_delta(x0)

    with open("LHDS.%i.txt" % bs_len, "w") as ofp:
        for train, test in zip(train_lhds, test_lhds):
            print >> ofp, train, test

    ref_energy, chem_pots, ddg_array = extract_data_from_array(x)
    test_lhd = calc_log_lhd(ref_energy, ddg_array, chem_pots, 0)

    return ddg_array, ref_energy, test_lhd
