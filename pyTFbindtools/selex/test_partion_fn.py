import os, sys
import math
from motif_tools import load_motifs, logistic, R, T, DeltaDeltaGArray, logit

from itertools import product, izip

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve

NBINS = 500

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

def est_partition_fn_sampling(ref_energy, ddg_array, n_bind_sites, seq_len):
    n_sims = PARTITION_FN_SAMPLE_SIZE
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


def bin_energies(energies, min_energy, max_energy, n_bins=NBINS):
    energy_range = max_energy - min_energy + 1e-12
    step_size = energy_range/(n_bins-1)
    hist_est = np.zeros(n_bins)
    n_seqs = 0
    for energy in energies:
        mean_bin = (energy-min_energy)/step_size
        lower_bin = int(mean_bin)
        upper_bin = int(np.ceil(mean_bin))
        a = upper_bin - mean_bin
        assert 0 <= lower_bin <= upper_bin <= n_bins
        hist_est[lower_bin] += a
        hist_est[upper_bin] += 1-a
        n_seqs += 1

    x = np.linspace(
        min_energy, min_energy+step_size*len(hist_est), len(hist_est));    
    return x, hist_est/n_seqs

def est_partition_fn_by_brute_force(motif, n_bins=NBINS):
    def iter_seqs():
        for i, seq in enumerate(product(*[(0,1,2,3)]*len(motif))):
            if i%10000 == 0: print i, 4**len(motif)
            yield motif.score_seq(seq)
    return bin_energies(iter_seqs(), motif.min_energy, motif.max_energy, n_bins)


def est_partition_fn(ref_energy, ddg_array, n_bins=NBINS):
    # reset the motif data so that the minimum value in each column is 0
    min_energy = ddg_array.calc_min_energy(ref_energy)
    max_energy = ddg_array.calc_max_energy(ref_energy)
    step_size = (max_energy-min_energy+1e-6)/(n_bins-ddg_array.motif_len)
    
    # build the polynomial for each base
    poly_sum = np.zeros(n_bins+1, dtype='float32')
    # for each bae, add to the polynomial
    for base_i, base_energies in enumerate(
            ddg_array.calc_base_contributions()):
        min_base_energy = base_energies.min()
        new_poly = np.zeros(
            1+np.ceil((base_energies.max()-min_base_energy)/step_size))
        
        for base_energy in base_energies:
            mean_bin = (base_energy-min_base_energy)/step_size
            lower_bin = int(mean_bin)
            upper_bin = int(np.ceil(mean_bin))

            a = upper_bin - mean_bin
            new_poly[lower_bin] += 0.25*a
            new_poly[upper_bin] += 0.25*(1-a)

        if base_i == 0:
            poly_sum[:len(new_poly)] = new_poly
        else:
            #poly_sum = np.convolve(poly_sum, new_poly)
            poly_sum = fftconvolve(poly_sum, new_poly)
    
    #assert n_bins+1 >= poly_sum.nonzero()[0].max()    
    poly_sum = poly_sum[:n_bins]

    x = np.linspace(min_energy, min_energy+step_size*len(poly_sum), len(poly_sum));
    assert len(x) == n_bins
    return x, poly_sum



def cmp_to_brute():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()
    x, part_fn = est_partition_fn(ref_energy, ddg_array, NBINS)
    min_cdf = 1 - (1 - part_fn.cumsum())**2
    min_pdf = np.array((min_cdf[1:] - min_cdf[:-1]).tolist() + [0.0,])
    x2, part_fn_brute = est_partition_fn_by_brute_force(motif, NBINS)

    part_plt, = plt.plot(x, part_fn, label='Part Fn')
    min_part_plt, = plt.plot(x, min_pdf, label='Min Part Fn')
    brute_plt, = plt.plot(x2, part_fn_brute, label='Brute Part Fn')
    plt.legend(handles=[part_plt, min_part_plt, brute_plt])
    plt.show()

    return

cmp_to_brute()
