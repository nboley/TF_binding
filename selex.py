import os, sys
import math
from motif_tools import (
    load_motifs, logistic, R, T, DeltaDeltaGArray, Motif, load_motif_from_text)

from itertools import product, izip, chain

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as TT

from scipy.optimize import (
    minimize, minimize_scalar, brentq, differential_evolution, anneal )
from numpy.fft import rfft, irfft

import random

import gzip

import warnings
warnings.simplefilter("ignore")

VERBOSE = False
DEBUG = False
CMP_LHD_NUMERATOR_CALCS = False
RANDOM_POOL_SIZE = None
CONVERGENCE_MAX_LHD_CHANGE = None
MAX_NUM_ITER = None

EXPECTED_MEAN_ENERGY = -3.0
CONSTRAIN_MEAN_ENERGY = True

CONSTRAIN_BASE_ENERGY_DIFF = True
MAX_BASE_ENERGY_DIFF = 8.0

# during optimization, how much to account for previous values
MOMENTUM = 0.5

"""
SELEX and massively parallel sequencing  
Sequence of the DNA ligand is described in Supplemental Table S3. The ligands 
contain all the sequence features necessary for direct sequencing using an 
Illumina Genome Analyzer. The ligands were synthesized from two single-stranded 
primers (Supplemental Table S3) using Taq polymerase. The 256 barcodes used 
consist of all possible 4-bp identifier sequences and a 1-bp 'checksum' 
nucleotide, which allows identification of most mutated sequences. The products 
bearing different barcodes can be mixed and later identified based on the unique
sequence barcodes.  
  
For SELEX, 50-100 ng of barcoded DNA fragments was added to the TF or 
DBD-containing wells in 50 uL of binding buffer containing 150-500 ng of 
poly(dI/dC)-oligonucleotide (Amersham 27-7875-01[discontinued] or Sigma 
P4929-25UN) competitor. The resulting molar protein-to-DNA and 
protein-to-binding site ratios are on the order of 1:25 and 1:15,000, 
respectively. The plate was sealed and mixtures were left to compete for 2 h 
in gentle shaking at room temperature. Unbound oligomers were cleared away from 
the plates by five rapid washes with 100-300 uL of ice-cold binding buffer. 
After the last washing step, the residual moisture was cleared by centrifuging 
the plate inverted on top of paper towels at 500g for 30 sec. The bound DNA was 
eluted into 50 uL of TE buffer (10 mM Tris-Cl at pH 8.0 containing 1 mM EDTA) 
by heating for 25 min to 85C, and the TE buffer was aspirated directly from 
the hot plate into a fresh 96-well storage plate.  
  
The efficiency of the SELEX was initially evaluated by real-time quantitative 
PCR (qPCR) on a Roche light cycler using the SYBR-green-based system and 
calculating the differences in eluted oligomer amount by crossing-point 
analysis. Seven microliters of eluate was amplified using PCR (19-25 cycles), 
and the products were used in subsequent cycles of SELEX. Nesting primers 
(Supplemental Table S3) moving at least 2 bp inward in each cycle were used to 
prevent amplification of contaminating products. For sequencing, approximately 
similar amounts of DNA from each sample were mixed to generate a multiplexed 
sample for sequencing. 
"""

#n_dna_seq = 7.5e-8/(1.02e-12*119) # molecules  - g/( g/oligo * oligo/molecule)
#dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
#prot_conc = dna_conc/25 # mol/L

# 50-100 1e-9 g DNA
# DNA sequence: TCCATCACGAATGATACGGCGACCACCGAACACTCTTTCCCTACACGACGCTCTTCCGATCTAAAATNNNNNNNNNNNNNNNNNNNNCGTCGTATGCCGTCTTCTGCTTGCCGACTCCG
# DNA is on average ~ 1.079734e-21 g/oligo * 119 oligos
# molar protein:DNA ratio: 1:25
# volume: 5.0e-5 L 
n_dna_seq = 7.5e-8/(1.079734e-21*119) # molecules  - g/( g/oligo * oligo/molecule)
dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
prot_conc = dna_conc/25 # mol/L (should be 25)
#prot_conc /= 1000

RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
base_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 0: 0, 1: 1, 2: 2, 3: 3}

def log(msg, level='NORMAL'):
    assert level in ('NORMAL', 'VERBOSE', 'DEBUG')
    if level == 'DEBUG' and not DEBUG: return 
    if level == 'VERBOSE' and not VERBOSE: return 
    print >> sys.stderr, msg

def base_map(base):
    if base == 'N':
        base = random.choice('ACGT')
    return base_map_dict[base]

def enumerate_binding_sites(seq, bs_len):
    for offset in xrange(0, len(seq)-bs_len+1):
        subseq = seq[offset:offset+bs_len].upper()
        yield subseq
        yield "".join(RC_map[base] for base in reversed(subseq))

def code_sequence(seq, motif_len):
    # store all binding sites (subseqs and reverse complements of length 
    # motif )
    coded_bss = []
    #coded_seq = np.array([base_map[base] for base in seq.upper()])
    for offset in xrange(0, len(seq)-motif_len+1):
        subseq = seq[offset:offset+motif_len]
        if isinstance(subseq, str): subseq = subseq.upper()
        # forward sequence
        coded_subseq = [
            pos*3 + (base_map(base) - 1) 
            for pos, base in enumerate(subseq)
            if base_map(base) != 0]
        coded_bss.append(np.array(coded_subseq, dtype=int))
        # reverse complement
        coded_subseq = [
            pos*3 + (2 - base_map(base)) 
            for pos, base in enumerate(reversed(subseq))
            if base_map(base) != 3]
        coded_bss.append(np.array(coded_subseq, dtype=int))

    return coded_bss

def code_seqs(seqs, motif_len, ON_GPU=True):
    """Load SELEX data and encode all the subsequences. 

    """
    subseq0 = code_sequence(seqs[0], motif_len)
    # find the sequence length
    seq_len = len(seqs[0])
    assert all( seq_len == len(seq) for seq in seqs )
    coded_seqs = np.zeros((len(seqs), len(subseq0), motif_len*3), 
                          dtype=theano.config.floatX)
    for i, seq in enumerate(seqs):
        for j, subseq in enumerate(code_sequence(seq, motif_len)):
            coded_seqs[i,j,subseq] = 1
    if ON_GPU:
        return theano.shared(coded_seqs)
    else:
        return coded_seqs

def load_text_file(fp):
    seqs = []
    for line in fp:
        seqs.append(line.strip().upper())
    return seqs

def load_fastq(fp):
    seqs = []
    for i, line in enumerate(fp):
        if i%4 == 1:
            seqs.append(line.strip().upper())
    return seqs

def est_partition_fn(ref_energy, ddg_array, n_bind_sites, n_bins=2**16):
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
        nonzero_bins = np.array(
            ((base_energies-base_energies.min())/step_size).round(), dtype=int)
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
        print "NUMERATORS", "="*10, numerators
        print chem_affinities
        print ref_energy
        print
    
    return numerators

def calc_lhd_denominators(ref_energy, ddg_array, chem_affinities, n_bind_sites):
    # now calculate the denominator (the normalizing factor for each round)
    # calculate the expected bin counts in each energy level for round 0
    energies, partition_fn = est_partition_fn(
        ref_energy, ddg_array, n_bind_sites)
    expected_cnts = (4**ddg_array.motif_len)*partition_fn
    curr_occupancies = np.ones(len(energies), dtype='float32')
    denominators = []
    for rnd, chem_affinity in enumerate(chem_affinities):
        curr_occupancies *= logistic(-(-chem_affinity+energies)/(R*T))
        denominators.append( np.log((expected_cnts*curr_occupancies).sum()))
    return denominators

def calc_log_lhd_factory(rnds_and_coded_seqs):
    n_bind_sites = rnds_and_coded_seqs[0].get_value().shape[1]
    
    calc_energy_fns = []
    sym_e = TT.vector()
    for x in rnds_and_coded_seqs:
        #calc_energy_fns.append(
        #    theano.function([sym_e], theano.sandbox.cuda.basic_ops.gpu_from_host(
        #        x.dot(sym_e).min(1)) ) )
        calc_energy_fns.append(
            theano.function([sym_e], x.dot(sym_e).min(1)) )
    
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     rnds_and_chem_affinities):
        assert len(rnds_and_coded_seqs) == len(rnds_and_chem_affinities)
        ref_energy = ref_energy.astype('float32')
        rnds_and_chem_affinities = rnds_and_chem_affinities.astype('float32')
        # score all of the sequences
        rnds_and_seq_ddgs = []
        for rnd, calc_energy in enumerate(calc_energy_fns):
            rnds_and_seq_ddgs.append( calc_energy(ddg_array) )
        # calculate the numerators
        numerators = calc_lhd_numerators(
            rnds_and_seq_ddgs, rnds_and_chem_affinities, ref_energy)

        # calcualte the denominators
        denominators = calc_lhd_denominators(
            ref_energy, ddg_array, rnds_and_chem_affinities, n_bind_sites)

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

def estimate_ddg_matrix(rnds_and_seqs, ddg_array, 
                            ref_energy, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    def f(x):
        x = x.astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            x, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, x, chem_pots)
        
        base_conts = x.calc_base_contributions()
        energy_diff = base_conts.max(1) - base_conts.min(1)
        penalty = (energy_diff[(energy_diff > 6)]**2).sum()

        print x.consensus_seq()
        print chem_pots
        print "Ref:", ref_energy
        print "Mean:", ref_energy + x.sum()/3
        print "Min:", x.calc_min_energy(ref_energy)
        print x.calc_base_contributions().round(2)
        print rv
        print penalty
        print rv - penalty + abs(ref_energy)

        return -rv + penalty

    x0 = ddg_array.astype('float32')
    lhd_0 = f(x0)
    res = minimize(f, x0, tol=ftol, method='Powell', # COBYLA  
                   options={'disp': False, 
                            'maxiter': 50000, 
                            'xtol': 1e-5, 'ftol': 1e-6} )
    if lhd_0 < f(res.x):
        return x0, f(res.x)
    return res.x.astype('float32').view(DeltaDeltaGArray), -f(res.x)

def estimate_dg_matrix_coord(rnds_and_seqs, ddg_array, 
                             ref_energy, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    def f_ddg(x, (index, ref_energy)):
        ddg_array[index] += x
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, ddg_array, chem_pots)
        
        ddg_array[index] -= x

        return -rv

    def f_ref_energy(x):
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy+x, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy+x, ddg_array, chem_pots)
        return -rv

    while True:
        initial_lhd = -f_ref_energy(0)
        res = minimize_scalar(f_ref_energy, bounds=[-5,5], tol=1e-3)  
        ref_energy += res.x
        for i, x0 in enumerate(ddg_array):
            res = minimize_scalar(
                f_ddg, bounds=[-5,5], args=[i,ref_energy], tol=1e-3)  
            ddg_array[i] += res.x
            print "="*50
            print i/3, i%3
            print ddg_array.consensus_seq()
            print "Ref:", ref_energy
            print est_chem_potentials(
                ddg_array, ref_energy, dna_conc, prot_conc,
                n_bind_sites, len(rnds_and_seqs))
            print "Mean:", ref_energy + ddg_array.sum()/3
            print "Min:", ddg_array.calc_min_energy(ref_energy)
            print ddg_array.calc_base_contributions().round(2)
            print res.fun
                
        if initial_lhd > -f_ref_energy(0) - 1e-6: break
        #print i, x0, res
    
    return ddg_array, ref_energy, -f_ref_energy(0)

def estimate_dg_matrix(rnds_and_seqs, ddg_array, 
                        ref_energy, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    def f_ddg(x, (base_pos, ref_energy)):
        ddg_array[3*base_pos:3*(base_pos+1)] += x
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, ddg_array, chem_pots)
        
        # Penalize models with non-physical mean affinities
        new_mean_energy = ref_energy + ddg_array.sum()/3
        if CONSTRAIN_MEAN_ENERGY or new_mean_energy > EXPECTED_MEAN_ENERGY:
            rv -= abs(new_mean_energy - EXPECTED_MEAN_ENERGY)**2
        
        # Penalize non-physical differences in base affinities
        if CONSTRAIN_BASE_ENERGY_DIFF:
            base_energy_diff = (
                max(0, ddg_array[3*base_pos:3*(base_pos+1)].max())
                - min(0, ddg_array[3*base_pos:3*(base_pos+1)].min()))
            if base_energy_diff > MAX_BASE_ENERGY_DIFF:
                rv -= (base_energy_diff)**2
        
        ddg_array[3*base_pos:3*(base_pos+1)] -= x
        return -rv

    def f_ref_energy(x):
        chem_pots = est_chem_potentials(
            ddg_array, ref_energy+x, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy+x, ddg_array, chem_pots)

        # Penalize models with non-physical mean affinities
        new_mean_energy = ref_energy + x + ddg_array.sum()/3
        if CONSTRAIN_MEAN_ENERGY or new_mean_energy > EXPECTED_MEAN_ENERGY:
            rv -= abs(new_mean_energy-EXPECTED_MEAN_ENERGY)**2

        return -rv

    iteration_number = 0
    prev_lhd = -1e100
    tol = 1e-2
    # initialize the lhd changes to a large number so each base is updated
    # once in the first round. We add an additional weight for the ref energy
    lhd_changes = 1e10*np.ones(ddg_array.motif_len+1)
    weights = lhd_changes/lhd_changes.sum()
    # until the minimum update tolerance is below the stop iteration threshold  
    while tol > CONVERGENCE_MAX_LHD_CHANGE and iteration_number < MAX_NUM_ITER:
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
            res = minimize_scalar(f_ref_energy, bounds=[-5,5], tol=tol)  
            new_lhd = -res.fun
            if new_lhd > prev_lhd:
                ref_energy += res.x
            else:
                print "STEP DIDNT IMPROVE THE LHD"
                new_lhd = prev_lhd
        # otherwise this is a base energy
        else:
            res = minimize(
                f_ddg, np.zeros(3, dtype=float),
                args=[base_index, ref_energy], 
                method='Powell', tol=tol)
            new_lhd = -res.fun
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
        # avoid a divide by zero
        lhd_changes += tol/10
        # don't update the base we just updated
        lhd_changes[base_index] = 0.0
        weights = lhd_changes/lhd_changes.sum()
        lhd_changes -= tol/10
        lhd_changes[base_index] = (
            lhd_changes[base_index]*(1-MOMENTUM) + MOMENTUM*(new_lhd-prev_lhd))
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
                        n_bind_sites, num_rnds):
    energy_grid, partition_fn = est_partition_fn(
        ref_energy, ddg_array, n_bind_sites)
    chem_pots = []
    for rnd in xrange(num_rnds):
        chem_pot = est_chem_potential(
            energy_grid, partition_fn,
            dna_conc, prot_conc )
        chem_pots.append(chem_pot)
        partition_fn *= logistic(-(-chem_pot+energy_grid)/(R*T))
        partition_fn = partition_fn/partition_fn.sum()
    return np.array(chem_pots, dtype='float32')

def generate_random_sequences( num, seq_len, bind_site_len  ):
    seqs = numpy.random.randint( 0, 4, num*seq_len ).reshape(num, seq_len)
    return parse_sequence_list( seqs, bind_site_len )

def simulate_reads( motif,
                    sim_sizes=(1000, 1000, 1000, 1000),
                    pool_size = 100000):
    ref_energy, ddg_array = motif.build_ddg_array()
    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, dna_conc, prot_conc, 2, len(sim_sizes))
    current_pool = np.array([np.random.randint(4, size=len(motif))
                             for i in xrange(pool_size)])
    rnds_and_seqs = []
    for rnd, (sim_size, chem_pot) in enumerate(
            zip(sim_sizes, chem_pots), start=1):
        occs = np.array([motif.est_occ(chem_pot, seq)
                         for seq in current_pool])
        #print current_pool
        seq_indices = np.random.choice(
            len(current_pool), size=sim_size,
            p=occs/occs.sum(), replace=True)
        seqs = current_pool[np.array(seq_indices, dtype=int)]
        seq_occs = occs[np.array(seq_indices, dtype=int)]

        with open("test_%s_rnd_%i.txt" % (motif.name, rnd), "w") as ofp:
            for seq in seqs:
                print >> ofp, "".join('ACGT'[x] for x in seq)
        current_pool = seqs[np.random.choice(
            len(seqs), size=pool_size,
            p=seq_occs/seq_occs.sum(), replace=True)]
        print "Finished simulations for round %i" % rnd
    
    # sys.argv[2]
    print "Finished Simulations"
    print "Ref Energy:", ref_energy
    print "Chem Pots:", chem_pots
    print ddg_array
    print "Waiting to continue..."
    raw_input()
    return
    #return

def load_sequences(fnames):
    rnds_and_seqs = []
    for fname in sorted(fnames,
                        key=lambda x: int(x.split("_")[-1].split(".")[0])):
        opener = gzip.open if fname.endswith(".gz") else open  
        with opener(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs.append( loader(fp) ) # [:1000]
    return rnds_and_seqs

def write_output(motif, ddg_array, ref_energy, ofp=sys.stdout):
    # normalize the array so that the consensus energy is zero
    consensus_energy = ddg_array.calc_min_energy(ref_energy)
    base_energies = ddg_array.calc_base_contributions()
    print >> ofp, ">%s.ENERGY\t%.6f" % (motif.name, consensus_energy)
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    conc_energies = []
    for pos, energies in enumerate(base_energies, start=1):
        conc_energies.append(
            energies - energies.min() - consensus_energy/len(base_energies))
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.6f" % (x - energies.min()) 
            for x in energies )

    print >> ofp, ">%s.PWM" % motif.name
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    for pos, energies in enumerate(conc_energies, start=1):
        pwm = 1-logistic(energies)
        pwm = pwm/pwm.sum()
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.4f" % x for x in pwm )

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
    log("Found consensus %imer '%s'" % (bs_len, consensus))
    counts = np.zeros((4, bs_len))
    for pos, base in enumerate(consensus): 
        counts[base_map(base), pos] += 1000

    # find a pwm using the initial sixmer alignment
    pwm = find_pwm_from_starting_alignment(rnds_and_seqs[0], counts)
    return pwm

def build_random_read_energies_pool(pool_size, read_len, ddg_array, ref_energy):
    energies = np.zeros(pool_size, dtype='float32')
    for i in xrange(pool_size):
        if i%10000 == 0: print "Bootstrapped %i reads." % i
        seq = np.random.randint(4, size=read_len)
        coded_seq = code_sequence(seq, ddg_array.motif_len)
        energy = 1e100
        for subseq in coded_seq:
            energy = min(energy, ddg_array[subseq].sum())
        energies[i] = energy
    return energies

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
        print "ENERGIES", rnd_seq_ddgs
        numerators = calc_lhd_numerators(
            rnd_seq_ddgs, chem_pots.astype('float32'), ref_energy) 
        print "NUMERATORS", "="*10, numerators
    
    return sum(numerators)

def bootstrap_lhds(read_len,
                   ddg_array, ref_energy, 
                   sim_sizes,
                   pool_size=10000 ):
    bs_len = ddg_array.motif_len
    # multiply vby two for the reverse complements
    n_bind_sites = 2*(read_len-bs_len+1)
    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, 
        dna_conc, prot_conc, 
        n_bind_sites,
        len(sim_sizes))

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
    initial_energy_pool = build_random_read_energies_pool(
        pool_size, read_len, ddg_array, ref_energy)
    lhds = []
    for i in xrange(100):
        numerator = bootstrap_lhd_numerators(
            initial_energy_pool, sim_sizes, 
            chem_pots, ref_energy, ddg_array)
        lhds.append( numerator - normalizing_constant )
    return lhds

def load_energy_data(fname):
    def load_energy(mo_text):
        lines = mo_text.strip().split("\n")
        motif_name, consensus_energy = lines[0].split()
        assert motif_name.endswith('.ENERGY')
        consensus_energy = float(consensus_energy)
        ddg_array = np.zeros((len(lines)-1,4))
        for pos, line in enumerate(lines[1:]):
            energies = np.array([float(x) for x in line.strip().split()[1:]])
            ddg_array[pos,:] = energies
        return consensus_energy, ddg_array
    
    with open(fname) as fp:
        models = fp.read().strip().split(">")
        # make sure there's a leading >
        assert models[0] == ''
        models = models[1:]
        assert len(models) == 2
        pwm_model_text = next(mo for mo in models if ".PWM" in mo.split("\n")[0])
        motif = load_motif_from_text(pwm_model_text)
        models.remove(pwm_model_text)
        assert len(models) == 1
        energy_mo_text = models[0]
        consensus_energy, ddg_array = load_energy(energy_mo_text)
        motif.update_energy_array(ddg_array, consensus_energy)
    
    return motif

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate energy models from a SELEX experiment.')

    parser.add_argument( '--selex-files', nargs='+', type=file, required=True,
                         help='Files containing SELEX reads.')

    parser.add_argument( '--background-sequence', type=file, 
        help='File containing reads sequenced from round 0.')

    parser.add_argument( '--starting-pwm', type=file,
                         help='A PWM to start from.')
    parser.add_argument( '--starting-energy-model', type=file,
                         help='An energy model to start from.')
    parser.add_argument( '--initial-binding-site-len', type=int, default=6,
        help='The starting length of the binding site (this will grow)')

    parser.add_argument( '--lhd-convergence-eps', type=float, default=1e-8,
                         help='Convergence tolerance for lhd change.')
    parser.add_argument( '--max-iter', type=float, default=1e5,
                         help='Maximum number of optimization iterations.')

    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e5,
        help='The random pool size for the bootstrap.')


    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    
    args = parser.parse_args()
    assert not (args.starting_pwm and args.starting_energy_model), \
            "Can not set both --starting-pwm and --starting-energy_model"

    global VERBOSE
    VERBOSE = args.verbose
    global RANDOM_POOL_SIZE
    RANDOM_POOL_SIZE = int(args.random_seq_pool_size)
    global CONVERGENCE_MAX_LHD_CHANGE
    CONVERGENCE_MAX_LHD_CHANGE = args.lhd_convergence_eps
    global MAX_NUM_ITER
    MAX_NUM_ITER = int(args.max_iter)
    
    if args.random_seed != None:
        np.random.seed(args.random_seed)
    

    log("Loading sequences", 'VERBOSE')
    rnds_and_seqs = load_sequences(x.name for x in args.selex_files)

    if args.starting_pwm != None:
        log("Loading PWM starting location", 'VERBOSE')
        motifs = load_motifs(args.starting_pwm)
        assert len(motifs) == 1, "Motif file contains multiple motifs"
        motif = motifs.values()[0]
        args.starting_pwm.close()
    elif args.starting_energy_model != None:
        log("Loading energy data", 'VERBOSE')
        motif = load_energy_data(args.starting_energy_model.name)
        args.starting_energy_model.close()
    else:
        log("Initializing starting location from %imer search" % args.initial_binding_site_len, 
            'VERBOSE')
        factor_name = 'TEST'
        bs_len = args.initial_binding_site_len
        pwm = find_pwm(rnds_and_seqs, args.initial_binding_site_len)
        motif = Motif("aligned_%imer" % args.initial_binding_site_len, 
                      factor_name, pwm)
    
    return motif, rnds_and_seqs

def main():
    motif, rnds_and_seqs = parse_arguments()
    ref_energy, ddg_array = motif.build_ddg_array()
    bs_len = ddg_array.motif_len
    log("Coding sequences", 'VERBOSE')
    coded_rnds_and_seqs = [ code_seqs(seqs, bs_len) 
                            for seqs in rnds_and_seqs ]

    opt_path = []
    lhd_hat = -1e50
    prev_lhd = -1e100
    ddg_array_hat = ddg_array.copy()
    ref_energy_hat = ref_energy
    while lhd_hat > prev_lhd + 1e-2:
        prev_lhd = lhd_hat
        ddg_array_hat, ref_energy_hat, lhd_hat = estimate_dg_matrix(
            coded_rnds_and_seqs, ddg_array_hat, ref_energy_hat)
        opt_path.append((ddg_array_hat, ref_energy_hat, lhd_hat))
        for i in xrange(10):
            print "="*100
        raw_input()
    for data in opt_path:
        print data
    
    # XXX this is messy, should be packaged into an object
    sim_sizes = [len(seqs) for seqs in rnds_and_seqs]
    read_len = len(rnds_and_seqs[0][0])
    n_bind_sites = read_len - bs_len + 1
    bs_lhds = bootstrap_lhds(
        read_len, ddg_array_hat, ref_energy_hat, 
        sim_sizes = sim_sizes,
        pool_size=RANDOM_POOL_SIZE )
    print "BS MEAN", np.array(bs_lhds).mean(), "SD", np.array(bs_lhds).std()
    print lhd_hat
        
    with open(motif.name + ".SELEX.txt", "w") as ofp:
        write_output(motif, ddg_array_hat, ref_energy_hat, ofp)
    
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
