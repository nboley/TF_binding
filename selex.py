import os, sys
import math
from motif_tools import load_motifs, logistic, R, T

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def generate_random_sequences( num, seq_len, bind_site_len  ):
    seqs = numpy.random.randint( 0, 4, num*seq_len ).reshape(num, seq_len)
    return parse_sequence_list( seqs, bind_site_len )

def load_selex_data():
    pass

def load_text_file(fname):
    seqs = []
    with open(fname) as fp:
        for line in fp:
            seqs.append(line.strip().upper())
    return seqs

def load_and_convert_text_file(fname):
    seqs = load_text_file(fname)
    seq_len = len(seqs[0])
    assert all( seq_len == len(seq) for seq in seqs )
    coded_seqs = []
    for i, seq in enumerate(seqs):
        coded_seq = np.array([base_map[base] for base in seq.upper()])
        coded_seqs.append(coded_seq)
    return np.vstack(coded_seqs)

def bin_energies(energies, min_energy, max_energy, n_bins=1000):
    energy_range = max_energy - min_energy + 1e-12
    step_size = energy_range/n_bins
    hist_est = np.zeros(n_bins+1)
    n_seqs = 0
    for energy in energies:
        mean_bin = (energy-min_energy)/step_size
        lower_bin = int(mean_bin)
        upper_bin = int(np.ceil(mean_bin))
        a = mean_bin - upper_bin
        assert 0 <= lower_bin <= upper_bin <= n_bins
        hist_est[lower_bin] += 1-a
        hist_est[upper_bin] += a
        n_seqs += 1
    
    x = np.linspace(min_energy, max_energy, n_bins+1);
    return x, hist_est/n_seqs

def score_seqs(energy_mat, coded_seqs):
    rv = np.zeros(len(coded_seqs), dtype=float)
    for seq_i, seq in enumerate(coded_seqs):
        rv[seq_i] = sum(energy_mat[i, base] for i, base in enumerate(seq))
    return rv

def calc_log_lhd(coded_seqs, motif, rnd, log_unbnd_conc):
    scored_seqs = score_seqs(motif.motif_data, coded_seqs)
    num = np.log(logistic(log_unbnd_conc+scored_seqs/(R*T))).sum()
    
    energies, partition_fn = est_partition_fn(motif)
    expected_cnts = (4**len(motif))*partition_fn
    occupancies = expected_cnts*(logistic(log_unbnd_conc+energies/(R*T)))
    denom = np.log(occupancies.sum())
    #print "====", num, len(coded_seqs)*denom, num - len(coded_seqs)*denom
    return num - len(coded_seqs)*denom

#def calc_log_lhd(seqs, motif, log_unbnd_conc):
#    log_lhd = 0
#    for seq in seqs:
#        log_lhd += math.log(motif.est_occ(log_unbnd_conc, seq))
#    return log_lhd

def est_partition_fn_by_sampling(motif, n_bins=1000, n_samples=1000000):
    energy_range = (motif.max_energy - motif.min_energy) + 1e-6
    hist_est = np.zeros(n_bins)
    for i in xrange(n_samples):
        if i%10000 == 0: print i, n_samples
        seq = np.random.randint(4, size=len(motif))
        shifted_energy = motif.score_seq(seq)-motif.consensus_energy
        bin = int(n_bins*shifted_energy/energy_range)
        assert 0 <= bin < n_bins
        hist_est[bin] += 1

    step_size = energy_range/n_bins
    x = np.arange(motif.min_energy, motif.max_energy + 1e-6, step_size);
    return x, hist_est

def est_partition_fn_by_brute_force(motif, n_bins=1000):
    def iter_seqs():
        for i, seq in enumerate(product(*[(0,1,2,3)]*len(motif))):
            if i%10000 == 0: print i, 4**len(motif)
            yield motif.score_seq(seq)-motif.consensus_energy
    return bin_energies(iter_seqs(), motif.min_energy, moitf.max_energy, n_bins)

def est_partition_fn(energy_matrix, n_bins=1000):
    min_energy = sum(min(x) for x in energy_matrix)
    max_energy = sum(max(x) for x in energy_matrix)
    step_size = (max_energy-min_energy)/n_bins
    
    # build the polynomial for each base
    poly_sum = np.zeros(n_bins+1, dtype=float)
    # for each bae, add to the polynomial
    for base_i, base_energies in enumerate(energy_matrix):
        new_poly = np.zeros(1+np.ceil(base_energies.max()/step_size))
        
        for base_energy in base_energies:
            mean_bin = base_energy/step_size
            lower_bin = int(mean_bin)
            upper_bin = int(np.ceil(mean_bin))
            a = 1 - (upper_bin - mean_bin )
            new_poly[lower_bin] += 0.25*a
            new_poly[upper_bin] += 0.25*(1-a)
            
        if base_i == 0:
            poly_sum[:len(new_poly)] = new_poly
        else:
            poly_sum = np.convolve(poly_sum, new_poly)
    
    #assert n_bins+1 >= poly_sum.nonzero()[0].max()    
    poly_sum = poly_sum[:n_bins+1]
    
    x = np.linspace(min_energy, max_energy, n_bins+1);
    return x, poly_sum


#def build_occs(energy_pdf, energies, rnd, chem_potential):
#    occ = logistic(chem_potential-energies)
#    occ_cumsum = occupancies.cumsum()**2
#    occ[:-1] = occ_cumsum[1:] - occ_cumsum[:-1]
#    # make sure this still sums to 1
#    occ[-1] += 1- occ.sum()
#    for i in xrange(rnd):
#        # raise occ to a power, which is equivalen
#        new_energies = occ*energies
#        pass
#    return occ*energies
        
def main():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    seqs = load_and_convert_text_file(sys.argv[2])

    energy_mat = motif.motif_data.copy()
    energy_mat[0,:] += motif.consensus_energy
    print energy_mat
    #assert False
    energies = score_seqs(energy_mat, seqs)
    min_energy = sum(min(x) for x in energy_mat)
    max_energy = sum(max(x) for x in energy_mat)
    x, binned_scores = bin_energies(energies, min_energy, max_energy, 1000)
    print "Binned Scores Sum", binned_scores.sum()
    x2, est_scores = est_partition_fn(energy_mat, 1000)
    print "Exp Scores Sum", binned_scores.sum()

    print x
    print x2
    return
    for chem_potential in np.linspace(-30, 20, 20):
        obs_occ = logistic(chem_potential-x/(R*T))*binned_scores
        n_obs_occ = obs_occ/obs_occ.sum()
        exp_occ = logistic(chem_potential-x2/(R*T))*est_scores
        n_exp_occ = exp_occ/exp_occ.sum()
        print "%.2f" % chem_potential, np.abs(n_obs_occ - n_exp_occ).sum()
    return
    """
    return
    x2, smart_hist_est = est_partition_fn(motif, 10000)
    weighted_ests = []

    """
    x2, smart_hist_est = est_partition_fn(motif, 10000)
    x, hist_est = est_partition_fn_by_brute_force(motif, 10000)

    weighted_ests = []
    chem_potential = -8.0
    for rnd in range(0,5):
        weighted_est = (logistic(chem_potential-x)**rnd)*hist_est
        print "Brute Occ:", weighted_est.sum(), 
        weighted_est /= weighted_est.sum()
        print (x*weighted_est).sum()
        weighted_ests.extend((x, weighted_est))

        #smt_weighted_est = build_occs(smart_hist_est, x2, rnd, chem_potential)
        smt_weighted_est = (logistic(chem_potential-x)**rnd)*smart_hist_est
        print "Smart Occ:",  smt_weighted_est.sum(),
        smt_weighted_est /= smt_weighted_est.sum()
        print (x2*smt_weighted_est).sum()
        weighted_ests.extend((x2, smt_weighted_est))

        #plt.scatter(weighted_est, smt_weighted_est)
        #plt.show()

        print "Motif Mean Energy", motif.mean_energy
        print 
    #plt.plot(*weighted_ests)
    #plt.show()

    pass

    #seqs = load_text_file(sys.argv[2])
    #for log_tf_conc in range(10):
    #    print log_tf_conc, calc_log_lhd(seqs, motif, log_tf_conc)
    
main()
