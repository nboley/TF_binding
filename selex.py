import os, sys
import math
from motif_tools import load_motifs, logistic, R, T, DeltaDeltaGArray, logit

from itertools import product, izip

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as TT

from scipy.optimize import minimize, leastsq, bisect, minimize_scalar
from scipy.signal import fftconvolve

import random

import gzip

VERBOSE = False

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
# DNA is ~ 1.02e-12 g/oligo * 119 oligos
# molar protein:DNA ratio: 1:25
# volume: 5.0e-5 L 
n_dna_seq = 7.5e-8/(1.02e-12*119) # molecules  - g/( g/oligo * oligo/molecule)
dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
prot_conc = dna_conc/25 # mol/L (should be 25)
#prot_conc /= 1000

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
    coded_seqs = np.zeros((len(seqs), len(subseq0), len(motif)*3), 
                          dtype=theano.config.floatX)
    print coded_seqs.shape
    for i, seq in enumerate(seqs):
        for j, subseq in enumerate(code_sequence(seq, motif)):
            coded_seqs[i,j,subseq] = 1
    #return coded_seqs
    return theano.shared(coded_seqs)

def est_partition_fn(ref_energy, ddg_array, n_bind_sites, n_bins=5000):
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
            # naive colvolve
            #convolve(poly_sum, new_poly)
            poly_sum = fftconvolve(poly_sum, new_poly, 'full')[:n_bins]

    poly_sum[poly_sum < 1e-16] = 0.0
    ## we lose this assert because we do this early
    #assert n_bins+1 >= poly_sum.nonzero()[0].max(), poly_sum.nonzero()[0].max()
    part_fn = poly_sum[:n_bins]

    min_cdf = 1 - (1 - part_fn.cumsum())**n_bind_sites
    min_pdf = np.array((min_cdf[1:] - min_cdf[:-1]).tolist() + [0.0,], dtype='float32')

    x = np.linspace(min_energy, min_energy+step_size*len(part_fn), len(part_fn));
    assert len(x) == n_bins
    return x, min_pdf

def calc_log_lhd_factory(rnds_and_coded_seqs):
    #sym_x = TT.vector('x')
    #calc_energy = theano.function([sym_x], theano.shared(coded_seqs).dot(sym_x))

    n_bind_sites = rnds_and_coded_seqs[0].get_value().shape[1]
    calc_energy_fns = []
    sym_e = TT.vector()
    for x in rnds_and_coded_seqs:
        #calc_energy_fns.append(
        #    theano.function([sym_e], theano.sandbox.cuda.basic_ops.gpu_from_host(
        #        x.dot(sym_e).min(1)) ) )
        calc_energy_fns.append(
            theano.function([sym_e], x.dot(sym_e).min(1)) )
    
    sym_cons_dg = TT.scalar('cons_dg')
    sym_chem_pot = TT.scalar('chem_pot')
    sym_ddg = TT.vector('ddg')
    
    calc_occ = theano.function([sym_chem_pot, sym_cons_dg, sym_ddg], -(
        TT.log(1.0 + TT.exp(
            (-sym_chem_pot + sym_cons_dg + sym_ddg)/(R*T))).sum()))

    
    #print calc_occ.maker.fgraph.toposort()
    #print
    #print calc_energy_fns[0].maker.fgraph.toposort()
    #assert False
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
        # the occupancies of each rnd are a function of the chemical affinity of
        # the round in which there were sequenced and each previous round. We 
        # loop through each sequenced round, and calculate the numerator of the log lhd 
        numerators = []
        for sequencing_rnd, seq_ddgs in enumerate(rnds_and_seq_ddgs):
            chem_affinity = rnds_and_chem_affinities[0]
            #numerator = np.log(logistic(-(-chem_affinity+ref_energy+seq_ddgs)/(R*T))).sum()
            numerator = calc_occ(chem_affinity, ref_energy, seq_ddgs)
            for rnd in xrange(1, sequencing_rnd+1):
                #numerator += np.log(
                #    logistic(-(-rnds_and_chem_affinities[rnd]+ref_energy+seq_ddgs)/(R*T))).sum()
                numerator += calc_occ(
                    rnds_and_chem_affinities[rnd], ref_energy, seq_ddgs).sum()
            numerators.append(numerator)

        # now calculate the denominator (the normalizing factor for each round)
        # calculate the expected bin counts in each energy level for round 0
        energies, partition_fn = est_partition_fn(
            ref_energy, ddg_array, n_bind_sites)
        expected_cnts = (4**ddg_array.motif_len)*partition_fn
        curr_occupancies = np.ones(len(energies), dtype='float32')
        denominators = []
        for rnd, chem_affinity in enumerate(rnds_and_chem_affinities):
            curr_occupancies *= logistic(-(-chem_affinity+energies)/(R*T))
            denominators.append( np.log((expected_cnts*curr_occupancies).sum()))

        lhd = 0.0
        for rnd_num, rnd_denom, rnd_seq_ddgs in izip(
                numerators, denominators, rnds_and_seq_ddgs):
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

def estimate_dg_matrix(rnds_and_seqs, ddg_array, ref_energy, chem_pots, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = len(rnds_and_seqs[0])
    def f(x):
        ref_energy = x[0]
        x = x[1:].astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            x, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, x, chem_pots)

        """
        print x.consensus_seq()
        print ref_energy
        print chem_pots
        print x.calc_min_energy(ref_energy)
        print x.calc_base_contributions()
        print rv
        """
        
        return -rv

    x0 = np.hstack((ref_energy, ddg_array))
    res = minimize(f, x0, tol=ftol, method='COBYLA',
                   options={'disp': False, 'maxiter': 10000},
                   bounds=[(-6,6) for i in xrange(len(x0))])
    return res.x[0], res.x[1:].view(DeltaDeltaGArray), -f(res.x)

def estimate_ddg_matrix(rnds_and_seqs, ddg_array, ref_energy, chem_pots, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]
    def f(x):
        x = x.astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            x, ref_energy, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(ref_energy, x, chem_pots)

        
        print x.consensus_seq()
        print ref_energy
        print chem_pots
        print x.calc_min_energy(ref_energy)
        print x.calc_base_contributions()
        print rv
        
        return -rv

    x0 = ddg_array
    res = minimize(f, x0, tol=ftol, method='COBYLA',
                   options={'disp': False, 'maxiter': 50000},
                   bounds=[(-6,6) for i in xrange(len(x0))])
    # x0 = res.x.view(DeltaDeltaGArray)
    #res = minimize(f, x0, tol=ftol, 
    #               options={'disp': False, 'maxiter': 10000},
    #               bounds=[(-6,6) for i in xrange(len(x0))])
    return res.x.view(DeltaDeltaGArray), -f(res.x)

def estimate_consensus_GFE(
        rnds_and_seqs, ddg_array, ref_energy, chem_pots, ftol=1e-12):
    calc_log_lhd = calc_log_lhd_factory(rnds_and_seqs)
    n_bind_sites = len(rnds_and_seqs[0])
    def f(x):
        #x = x.astype('float32').view(DeltaDeltaGArray)
        chem_pots = est_chem_potentials(
            ddg_array, x, dna_conc, prot_conc,
            n_bind_sites, len(rnds_and_seqs))
        rv = calc_log_lhd(x, ddg_array, chem_pots)

        print chem_pots
        print x, rv
        print
        return -rv

    for x in np.linspace(-50, 5, 100):
        f(x)
    assert False
    x0 = ref_energy
    res = minimize_scalar(f,bounds=[-50,-20],method='bounded')
    assert False
    return 


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
    rv = bisect(f, min_u, max_u, xtol=1e-4)
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

def simulations(motif):
    pool_size = 100000
    sim_sizes = [1000, 1000, 1000, 1000]

    n_dna_seq = 7.5e-8/(1.02e-12*119) # g/molecule  - g/( g/oligo * oligo/molecule)
    dna_conc = 6.02e23*n_dna_seq/5.0e-5 # mol/L
    prot_conc = dna_conc/25 # mol/L

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

def load_sequences(fnames, motif):
    rnds_and_seqs = []
    for fname in sorted(fnames,
                        key=lambda x: int(x.split("_")[-1].split(".")[0])):
        opener = gzip.open if fname.endswith(".gz") else open  
        with opener(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            coded_seqs = code_seqs_as_matrix(loader(fp), motif)
            rnds_and_seqs.append( coded_seqs )
    print "Finished loading sequences"
    return rnds_and_seqs

def main():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()
    #simulations(motif)
    #return
    
    rnds_and_seqs = load_sequences(sys.argv[2:], motif)
    n_bind_sites = rnds_and_seqs[0].get_value().shape[1]

    motif = load_motifs(sys.argv[1]).values()[0][0]
    ref_energy, ddg_array = motif.build_ddg_array()

    chem_pots = est_chem_potentials(
        ddg_array, ref_energy,
        dna_conc, prot_conc,
        n_bind_sites, len(rnds_and_seqs))
    print ddg_array.consensus_seq()
    print chem_pots
    print ref_energy
    print ddg_array.calc_min_energy(ref_energy)
    print ddg_array.calc_base_contributions()
    print 
    print "Finished loading motif"

    ref_energy = np.float32(-2.0)
    x = ddg_array
    #x = np.random.uniform(size=len(ddg_array)).view(DeltaDeltaGArray)
    chem_pots = est_chem_potentials(
        x, ref_energy, dna_conc, prot_conc, n_bind_sites, len(rnds_and_seqs))
    print "Chem Pots:", chem_pots
    #raw_input()
    x, lhd = estimate_ddg_matrix(
        rnds_and_seqs, x, ref_energy, chem_pots)
    print x.consensus_seq()
    print est_chem_potentials(
        x, ref_energy, dna_conc, prot_conc, n_bind_sites, len(rnds_and_seqs))
    print ref_energy
    print x.calc_min_energy(ref_energy)
    print x.calc_base_contributions()
    print lhd

    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    
    return

main()
