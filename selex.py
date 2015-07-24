import os, sys
import math
from motif_tools import load_motifs, logistic, R, T, DeltaDeltaGArray, logit

from itertools import product, izip

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as TT

from scipy.optimize import minimize, leastsq, bisect

import random

import gzip

VERBOSE = False

n_dna_seq = 7.5e-8/(1.02e-12*119) # g/molecule  - g/( g/oligo * oligo/molecule)
dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
prot_conc = dna_conc/25 # mol/L

base_map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
def base_map(base):
    if base == 'N':
        base = random.choice('ACGT')
    return base_map_dict[base]

def generate_random_sequences( num, seq_len, bind_site_len  ):
    seqs = numpy.random.randint( 0, 4, num*seq_len ).reshape(num, seq_len)
    return parse_sequence_list( seqs, bind_site_len )

def code_sequence(seq, motif):
    # store all binding sites (subseqs and reverse complements of length 
    # motif )
    coded_bss = []
    #coded_seq = np.array([base_map[base] for base in seq.upper()])
    for offset in xrange(0, len(seq)-len(motif)+1):
        subseq = seq[offset:offset+len(motif)].upper()
        # forward sequence
        coded_subseq = [
            pos*3 + (base_map(base) - 1) 
            for pos, base in enumerate(subseq)
            if base != 'A']
        coded_bss.append(np.array(coded_subseq, dtype=int))
        # reverse complement
        coded_subseq = [
            pos*3 + (2 - base_map(base)) 
            for pos, base in enumerate(reversed(subseq))
            if base != 'T']
        coded_bss.append(np.array(coded_subseq, dtype=int))

    return coded_bss

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

def code_seqs(seqs, motif):
    """Load SELEX data and encode all the subsequences. 

    """
    # find the sequence length
    seq_len = len(seqs[0])
    assert all( seq_len == len(seq) for seq in seqs )
    coded_seqs = []
    for i, seq in enumerate(seqs):
        coded_seqs.append(code_sequence(seq, motif))
    return coded_seqs

def code_seqs_as_matrix(seqs, motif):
    """Load SELEX data and encode all the subsequences. 

    """
    subseq0 = code_sequence(seqs[0], motif)
    # find the sequence length
    seq_len = len(seqs[0])
    assert all( seq_len == len(seq) for seq in seqs )
    coded_seqs = [ np.zeros((len(seqs), len(motif)*3), dtype='float32')
                   for i in xrange(len(subseq0)) ]
    for i, seq in enumerate(seqs):
        for j, subseq in enumerate(code_sequence(seq, motif)):
            coded_seqs[j][i,subseq] = 1
    return coded_seqs
    return theano.shared(coded_seqs, theano.config.floatX)

def est_partition_fn(ref_energy, ddg_array, n_bins=5000):
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
            1+np.ceil((base_energies.max()-min_base_energy)/step_size),
            dtype='float32')
        
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
    part_fn = poly_sum[:n_bins]

    min_cdf = 1 - (1 - part_fn.cumsum())**20
    min_pdf = np.array((min_cdf[1:] - min_cdf[:-1]).tolist() + [0.0,], dtype='float32')

    x = np.linspace(min_energy, min_energy+step_size*len(part_fn), len(part_fn));
    assert len(x) == n_bins
    return x, min_pdf

def calc_log_lhd_factory(rnds_and_coded_seqs):
    """
    sym_x = TT.vector('x')
    calc_energy = theano.function([sym_x], theano.shared(coded_seqs).dot(sym_x))
    """
    sym_e = TT.vector('e')
    sym_c = TT.scalar('c')
    f = theano.function([sym_e, sym_c], (
        -TT.log(1.0 + TT.exp(-(sym_c - sym_e)/(R*T)))).sum())

    """
    print f.maker.fgraph.toposort()
    if np.any([isinstance(x.op, TT.Elemwise) for x in f.maker.fgraph.toposort()]):
        print 'Used the cpu'
    else:
        print 'Used the gpu'
    """
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     rnds_and_chem_affinities):
        assert len(rnds_and_coded_seqs) == len(rnds_and_chem_affinities)

        # score all of the sequences
        rnds_and_seq_ddgs = []
        for rnd, coded_seqs in enumerate(rnds_and_coded_seqs):
            rnd_energies = np.vstack(
                [coded_subseqs.dot(ddg_array)
                 for coded_subseqs in coded_seqs]).min(0)
            rnds_and_seq_ddgs.append( rnd_energies )
        # the occupancies of each rnd are a function of the chemical affinity of
        # the round in which there were sequenced and each previous round. We 
        # loop through each sequenced round, and calculate the numerator of the log lhd 
        numerators = []
        for sequencing_rnd, seq_ddgs in enumerate(rnds_and_seq_ddgs):
            chem_affinity = rnds_and_chem_affinities[0]
            #numerator = np.log(logistic((chem_affinity-ref_energy-seq_ddgs)/(R*T))).sum()
            numerator = f(seq_ddgs, (chem_affinity-ref_energy).astype('float32'))
            for rnd in xrange(1, sequencing_rnd+1):
                #numerator += np.log(
                #    logistic((rnds_and_chem_affinities[rnd]-ref_energy-seq_ddgs)/(R*T))).sum()
                numerator += f(
                    seq_ddgs, (rnds_and_chem_affinities[rnd]-ref_energy).astype('float32')).sum()
            numerators.append(numerator)

        # now calculate the denominator (the normalizing factor for each round)
        # calculate the expected bin counts in each energy level for round 0
        energies, partition_fn = est_partition_fn(ref_energy, ddg_array)
        expected_cnts = (4**ddg_array.motif_len)*partition_fn
        curr_occupancies = np.ones(len(energies), dtype='float32')
        denominators = []
        for rnd, chem_affinity in enumerate(rnds_and_chem_affinities):
            curr_occupancies *= logistic((chem_affinity-energies)/(R*T))
            denominators.append( np.log((expected_cnts*curr_occupancies).sum()))

        lhd = 0.0
        """
        print "+"*40
        print numerators
        print denominators
        print rnds_and_seq_ddgs
        print "="*40
        """
        for rnd_num, rnd_denom, rnd_seq_ddgs in izip(
                numerators, denominators, rnds_and_seq_ddgs):
            #print rnd_num, len(rnd_seq_ddgs), rnd_denom
            lhd += rnd_num - len(rnd_seq_ddgs)*rnd_denom

        return lhd
    
    return calc_log_lhd

def iter_simulated_seqs(motif, chem_pots):
    cnt = 0
    seqs = []
    while True:
        seq = np.random.randint(4, size=len(motif))
        occ = 1.0
        for chem_pot in chem_pots:
            occ *= motif.est_occ(chem_pot, seq)
        if random.random() < occ:
            yield seq, occ
    return
        
def sim_seqs(ofname, n_seq, motif, chem_pots):
    fp = open(ofname, "w")
    for i, (seq, occ) in enumerate(
            iter_simulated_seqs(motif, chem_pots)):
        print >> fp, "".join('ACGT'[x] for x in seq)
        if i >= n_seq: break
        if i%100 == 0: print "Finished sim seq %i/%i" % (i, n_seq)
    fp.close()
    return

def test():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()

    chem_pots = [-6, -7, -8, -9]
    rnds_and_seqs = []
    sim_sizes = [100, 100, 100, 100]
    for rnd, (sim_size, chem_pot) in enumerate(
            zip(sim_sizes, chem_pots), start=1):
        if sim_size == 0:
            rnds_and_seqs.append([])
        else:
            ofname = "test.rnd%i.cp%.2e.txt" % (rnd, chem_pot)
            #sim_seqs(ofname, sim_size, motif, chem_pots[:rnd]) 
            rnds_and_seqs.append( load_and_code_text_file(ofname, motif) )
    # sys.argv[2]
    print "Finished Simulations"
    #return

    print ddg_array.consensus_seq()
    print ref_energy
    print ddg_array.calc_min_energy(ref_energy)
    print ddg_array.calc_base_contributions()

    #print calc_rnd_log_lhd(
    #    rnds_and_seqs[-1], ref_energy, ddg_array, len(chem_pots), chem_pots[-1])
    print calc_log_lhd(rnds_and_seqs, ref_energy, ddg_array, chem_pots)
    
    log_chem_pot = -8
    def f(x):
        x = x.view(DeltaDeltaGArray)
        #rv = calc_rnd_log_lhd(seqs, ref_energy, x, rnd, log_chem_pot)
        rv = calc_log_lhd(rnds_and_seqs, ref_energy, x, chem_pots)
        if VERBOSE:
            print x.consensus_seq()
            print ref_energy
            print x.calc_min_energy(ref_energy)
            print x.calc_base_contributions()
            print rv
            print
        return -rv

    x0 = np.array([random.random() for i in xrange(len(ddg_array))], dtype='float32')
    # user a slow buty safe algorithm to find a starting point
    #res = minimize(f, x0, tol=1e-2,
    #               options={'disp': True, 'maxiter': 5000}
    #               , method='Powell') #'Nelder-Mead')
    #print "Finished finding a starting point" 
    res = minimize(f, x0, tol=1e-6,
                   options={'disp': True, 'maxiter': 50000},
                   bounds=[(-6,6) for i in xrange(len(x0))])
    global VERBOSE
    VERBOSE = True
    f(res.x)
    
    f(ddg_array)
    return

def estimate_ddg_matrix(rnds_and_seqs, ddg_array, ref_energy, chem_pots, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    
    def f(x):
        ref_energy = x[0]
        x = x[1:].astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            x, ref_energy, dna_conc, prot_conc, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, x, chem_pots)

        print x.consensus_seq()
        print ref_energy
        print chem_pots
        print x.calc_min_energy(ref_energy)
        print x.calc_base_contributions()
        print rv

        
        return -rv

    x0 = np.hstack((ref_energy, ddg_array))
    res = minimize(f, x0, tol=ftol, method='COBYLA',
                   options={'disp': False, 'maxiter': 10000},
                   bounds=[(-6,6) for i in xrange(len(x0))])
    return res.x[0], res.x[1:].view(DeltaDeltaGArray), -f(res.x)

def estimate_chem_pots_lhd(rnds_and_seqs, ddg_array, ref_energy, chem_pots):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)

    if VERBOSE:
        print ddg_array.consensus_seq()
        print ref_energy
        print ddg_array.calc_min_energy(ref_energy)
        print ddg_array.calc_base_contributions()
        print calc_log_lhd(ref_energy, ddg_array, chem_pots)

    def f(x):
        rv = calc_log_lhd(ref_energy, ddg_array, x)
        if VERBOSE:
            print rv, x
        return -rv

    x0 = chem_pots 
    res = minimize(f, x0, tol=1e-12, method='COBYLA',
                   options={'disp': False, 'maxiter': 50000},
                   bounds=[(-30,0) for i in xrange(len(x0))])
    return res.x

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
    min_u = -100
    max_u = math.log(prot_conc)
    rv = bisect(f, min_u, max_u, xtol=1e-8)
    return rv

def est_chem_potentials(ddg_array, ref_energy, dna_conc, prot_conc, num_rnds):
    energy_grid, partition_fn = est_partition_fn(ref_energy, ddg_array)
    chem_pots = []
    for rnd in xrange(num_rnds):
        chem_pot = est_chem_potential(
            energy_grid, partition_fn,
            dna_conc, prot_conc )
        chem_pots.append(chem_pot)
        partition_fn *= logistic((chem_pot-energy_grid)/(R*T))
        partition_fn = partition_fn/partition_fn.sum()
    return np.array(chem_pots, dtype='float32')

def simulations(motif):
    sim_sizes = [100, 100, 100, 100]

    n_dna_seq = 7.5e-8/(1.02e-12*119) # g/molecule  - g/( g/oligo * oligo/molecule)
    dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
    prot_conc = dna_conc/25 # mol/L

    ref_energy, ddg_array = motif.build_ddg_array()
    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, dna_conc, prot_conc, len(sim_sizes))

    rnds_and_seqs = []
    for rnd, (sim_size, chem_pot) in enumerate(
            zip(sim_sizes, chem_pots), start=1):
        if sim_size == 0:
            rnds_and_seqs.append([])
        else:
            ofname = "test_%s_rnd_%i.txt" % (motif.name, rnd)
            sim_seqs(ofname, sim_size, motif, chem_pots[:rnd]) 
    # sys.argv[2]
    print "Finished Simulations"
    return
    #return

    
def main():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()
    #simulations(motif)
    #return

    # 50-100 1e-9 g DNA
    # DNA sequence: TCCATCACGAATGATACGGCGACCACCGAACACTCTTTCCCTACACGACGCTCTTCCGATCTAAAATNNNNNNNNNNNNNNNNNNNNCGTCGTATGCCGTCTTCTGCTTGCCGACTCCG
    # DNA is ~ 1.02e-12 g/oligo * 119 oligos
    # molar protein:DNA ratio: 1:25
    # volume: 5.0e-5 L 
    n_dna_seq = 7.5e-8/(1.02e-12*119) # g/molecule  - g/( g/oligo * oligo/molecule)
    dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
    prot_conc = dna_conc/25 # mol/L

    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()
    print "Finished loading motif"
    
    rnds_and_seqs = []
    for fname in sorted(sys.argv[2:],
                        key=lambda x: int(x.split("_")[-1].split(".")[0])):
        with gzip.open(fname) as fp:
            coded_seqs = code_seqs_as_matrix(load_fastq(fp), motif)
            #coded_seqs = code_seqs_as_matrix(load_text_file(fp), motif)
            rnds_and_seqs.append( coded_seqs )
    print "Finished loading sequences"

    x = ddg_array
    #x = np.random.uniform(size=len(ddg_array)).view(DeltaDeltaGArray)
    chem_pots = est_chem_potentials(
        x, ref_energy, dna_conc, prot_conc, len(rnds_and_seqs))
    print "Chem Pots:", chem_pots
    #raw_input()
    ref_energy, x, lhd = estimate_ddg_matrix(
        rnds_and_seqs, x, ref_energy, chem_pots)
    print x.consensus_seq()
    print ref_energy
    print x.calc_min_energy(ref_energy)
    print x.calc_base_contributions()
    print lhd
    return


    for i in xrange(12):
        chem_pots = est_chem_potentials(
            x, ref_energy, dna_conc, prot_conc, len(rnds_and_seqs))
        print "Chem Pots:", chem_pots
        #raw_input()
        x, lhd = estimate_ddg_matrix(
            rnds_and_seqs, x, ref_energy, chem_pots, ftol=10**(-i))
        print x.consensus_seq()
        print ref_energy
        print x.calc_min_energy(ref_energy)
        print x.calc_base_contributions()
        print lhd

        
    return
    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, 
        dna_conc, prot_conc, 
        len(rnds_and_seqs))
    print chem_pots

    return

main()
