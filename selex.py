import os, sys
import math
from motif_tools import load_motifs, logistic, R, T, EnergyArray

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, leastsq

import random

VERBOSE = True

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def generate_random_sequences( num, seq_len, bind_site_len  ):
    seqs = numpy.random.randint( 0, 4, num*seq_len ).reshape(num, seq_len)
    return parse_sequence_list( seqs, bind_site_len )

def load_text_file(fname):
    seqs = []
    with open(fname) as fp:
        for line in fp:
            seqs.append(line.strip().upper())
    return seqs

def load_and_code_text_file(fname, motif):
    """Load SELEX data and encode all the subsequences. 

    """
    seqs = load_text_file(fname)
    # find the sequence length
    seq_len = len(seqs[0])
    assert all( seq_len == len(seq) for seq in seqs )
    coded_seqs = []
    for i, seq in enumerate(seqs):
        # store all binding sites (subseqs and reverse complements of length 
        # motif )
        coded_bss = []
        #coded_seq = np.array([base_map[base] for base in seq.upper()])
        for offset in xrange(0, seq_len-len(motif)+1):
            subseq = seq[offset:offset+len(motif)].upper()
            coded_subseq = [
                1 + pos*3 + base_map[base] - 1 
                for pos, base in enumerate(subseq)
                if base != 'A']
            coded_bss.append(np.array(coded_subseq))
        coded_seqs.append(coded_bss)
    return coded_seqs


def bin_energies(energies, min_energy, max_energy, n_bins=1000):
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

    x = np.linspace(min_energy, step_size*len(hist_est), len(hist_est));    
    return x, hist_est/n_seqs

def bin_energies_from_grid(energies, energy_grid):
    n_bins = len(energy_grid)
    min_energy = energy_grid[0]
    max_energy = energy_grid[-1]
    step_size = (max_energy - min_energy)/n_bins
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

    return hist_est #/n_seqs


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
            yield motif.score_seq(seq)
    return bin_energies(iter_seqs(), motif.min_energy, motif.max_energy, n_bins)

def est_partition_fn_orig(energy_matrix, n_bins=1000):
    # reset the motif data so that the minimum value in each column is 0
    min_energy = sum(min(x) for x in energy_matrix)
    max_energy = sum(max(x) for x in energy_matrix)
    step_size = (max_energy-min_energy+1e-6)/(n_bins-energy_matrix.shape[0])
    
    # build the polynomial for each base
    poly_sum = np.zeros(n_bins+1, dtype=float)
    # for each bae, add to the polynomial
    for base_i, base_energies in enumerate(energy_matrix):
        min_base_energy = base_energies.min()
        new_poly = np.zeros(
            1+np.ceil((base_energies.max()-min_base_energy)/step_size))
        
        for base_energy in base_energies:
            mean_bin = (base_energy-min_base_energy)/step_size
            lower_bin = int(mean_bin)
            upper_bin = int(np.ceil(mean_bin))
            a = mean_bin - upper_bin
            new_poly[lower_bin] += 0.25*a
            new_poly[upper_bin] += 0.25*(1-a)

        if base_i == 0:
            poly_sum[:len(new_poly)] = new_poly
        else:
            poly_sum = np.convolve(poly_sum, new_poly)
    
    assert n_bins+1 >= poly_sum.nonzero()[0].max()    
    poly_sum = poly_sum[:n_bins+1]
    
    x = np.linspace(0, max_energy, step_size);
    return x, poly_sum

def est_partition_fn(energy_array, n_bins=1000):
    # reset the motif data so that the minimum value in each column is 0
    min_energy = energy_array.calc_min_energy()
    max_energy = energy_array.calc_max_energy()
    step_size = (max_energy-min_energy+1e-6)/(n_bins-energy_array.motif_len)
    
    # build the polynomial for each base
    poly_sum = np.zeros(n_bins+1, dtype=float)
    # for each bae, add to the polynomial
    for base_i, base_energies in enumerate(
            energy_array.calc_base_contributions()):
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
            poly_sum = np.convolve(poly_sum, new_poly)
    
    assert n_bins+1 >= poly_sum.nonzero()[0].max()    
    poly_sum = poly_sum[:n_bins]
    
    x = np.linspace(min_energy, min_energy+step_size*len(poly_sum), len(poly_sum));
    assert len(x) == n_bins
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

def cmp_to_brute():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    energy_array = motif.build_energy_array()
    x, part_fn = est_partition_fn(energy_array, 256)
    x2, part_fn_brute = est_partition_fn_by_brute_force(motif, 1000)
    plt.plot(*[x, part_fn, x2, part_fn_brute])
    plt.show()
    return

def calc_rnd_log_lhd(coded_seqs, energy_array, rnd, log_unbnd_conc):
    scored_seqs = np.zeros(len(coded_seqs), dtype=float)
    for i, subseqs in enumerate(coded_seqs):
        scored_seqs[i] = max(energy_array[subseq].sum() for subseq in subseqs)
        
    numerator = rnd*np.log(logistic((log_unbnd_conc-scored_seqs)/(R*T))).sum()
    
    energies, partition_fn = est_partition_fn(energy_array)
    expected_cnts = (4**energy_array.motif_len)*partition_fn
    occupancies = logistic((log_unbnd_conc-energies)/(R*T))**rnd
    denom_occupancies = expected_cnts*occupancies
    denom = np.log(denom_occupancies.sum())
    #print "====", numerator, len(coded_seqs)*denom, 
    return numerator - len(coded_seqs)*denom


def calc_l2_loss(coded_seqs, energy_array, rnd, log_unbnd_conc):
    # build the partition function
    energy_grid, partition_fn = est_partition_fn(
        energy_array, n_bins=50*int(math.sqrt(len(coded_seqs))))
    expected_occupancies = logistic((log_unbnd_conc-energy_grid)/(R*T))**(rnd+1)
    expected_occupancies = expected_occupancies/expected_occupancies.sum()
    expected_cnts = expected_occupancies*len(coded_seqs)
    
    # score and bin the sequences
    scored_seqs = np.zeros(len(coded_seqs), dtype=float)
    for i, subseqs in enumerate(coded_seqs):
        scored_seqs[i] = min(energy_array[subseq].sum() for subseq in subseqs)
    binned_seq_scores = bin_energies_from_grid(
        scored_seqs, energy_grid)
    
    #print energy_grid
    #print expected_cnts
    #print binned_seq_scores
    # return the loss
    return expected_cnts - binned_seq_scores
    return ((expected_cnts - binned_seq_scores)**2).sum()

def iter_simulated_seqs(motif, rnd, tf_conc):
    cnt = 0
    seqs = []
    while True:
        seq = np.random.randint(4, size=len(motif))
        occ = motif.est_occ(tf_conc, seq)**rnd
        if random.random() < occ:
            yield seq, occ
    return
        
def sim_seqs(ofname, n_seq, motif, log_chem_pot, rnd):
    fp = open(ofname, "w")
    for i, (seq, occ) in enumerate(
            iter_simulated_seqs(motif, rnd, log_chem_pot)):
        print >> fp, "".join('ACGT'[x] for x in seq)
        if i >= n_seq: break
        if i%100 == 0: print "Finished sim seq %i/%i" % (i, n_seq)
    fp.close()
    return

def opt_l2_loss():
    def f(x, log_chem_pot):
        #x = np.clip(x, -20, 10)
        #x_w_en = np.array([energy_array[0],] + x.tolist()).view(EnergyArray)
        x_w_en = x.view(EnergyArray)
        rv = calc_l2_loss(seqs, x_w_en, rnd, log_chem_pot)
        if VERBOSE:
            print x_w_en.consensus_seq()
            print x_w_en[0]
            print x_w_en.calc_min_energy()
            print x_w_en.calc_base_contributions()
            print (rv**2).sum()
            print
        return rv

    log_chem_pot = -2
    x0 = np.array([random.random() for i in xrange(len(energy_array))])
    #x0 = energy_array
    fit_array, fit_cov, info, mesg, status = leastsq(
        f, x0, args=(log_chem_pot,), 
        full_output=True, factor=0.1, maxfev=1000*len(x0)) # 10*
    fit_array = fit_array.view(EnergyArray)
    print fit_array.consensus_seq()
    print fit_array.calc_min_energy()
    print fit_array[0]
    print fit_array.calc_base_contributions()
    print "Conv Status: ", mesg
    #print info
    #leastsq(f, energy_array, 
    #        options={'disp': True},
    #        method='powell') # Nelder-Mead
    return


def calc_log_lhd():
    pass

def main():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    energy_array = motif.build_energy_array()
    log_chem_pot = -8
    rnd = 1
    
    ofname = "test.rnd%i.cp%.2e.txt" % (rnd, log_chem_pot)
    #sim_seqs(ofname, 1000, motif, log_chem_pot, rnd) 
    seqs = load_and_code_text_file(ofname, motif)
    # sys.argv[2]
    print "Finished Simulations"
    #return

    print energy_array.consensus_seq()
    print energy_array[0]
    print energy_array.calc_min_energy()
    print energy_array.calc_base_contributions()
    
    log_chem_pot = -8
    all_A_energy = energy_array[0]
    def f(x):
        x_w_en = np.insert(x,0,all_A_energy).view(EnergyArray)
        rv = calc_rnd_log_lhd(seqs, x_w_en, rnd, log_chem_pot)

        if VERBOSE:
            print x_w_en.consensus_seq()
            print x_w_en[0]
            print x_w_en.calc_min_energy()
            print x_w_en.calc_base_contributions()
            print rv
            print
        return -rv

    x0 = np.array([random.random() for i in xrange(len(energy_array))])[1:]    
    # user a slow buty safe algorithm to find a starting point
    #res = minimize(f, x0, tol=1e-2,
    #               options={'disp': True, 'maxiter': 5000}
    #               , method='Powell') #'Nelder-Mead')
    #print "Finished finding a starting point" 
    res = minimize(f, x0, tol=1e-12,
                   options={'disp': True, 'maxiter': 50000},
                   bounds=[(-6,6) for i in xrange(len(x0))])
    global VERBOSE
    VERBOSE = True
    f(res.x)
    
    f(energy_array[1:])
    return
    #for chem_potential in np.linspace(-30, 20, 20):
    #    print chem_potential, calc_rnd_log_lhd(
    #        seqs, energy_array, 4, chem_potential)

    #print energy_array[1:].reshape(((len(energy_array)-1)/3,3))
    return
    consensus_energy = motif.consensus_energy
    energy_mat = motif.motif_data
    x2, est_scores = est_partition_fn(energy_mat, 1000)
    print est_scores.sum()
    return
    
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
