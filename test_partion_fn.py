import os, sys
import math
from motif_tools import load_motifs, logistic, R, T, DeltaDeltaGArray, logit

from itertools import product, izip

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve

NBINS = 500

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
