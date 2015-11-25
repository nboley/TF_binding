import os, sys

import math

import numpy as np

sys.path.insert(0, "/home/nboley/src/TF_binding/")

from scipy.optimize import brentq

import pyTFbindtools
from pyTFbindtools.motif_tools import load_energy_data, load_motifs

from pyDNAbinding.DB import load_binding_models_from_db
from pyDNAbinding.sequence import sample_random_seqs
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ

T = 300
R = 1.987e-3 # in kCal/mol*K

def est_chem_potential(seqs, binding_model, dna_conc, prot_conc):
    """Estimate chemical affinity for round 1.

    [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
    exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
    """
    # calculate the binding affinities for each sequence
    affinities = -(seqs.score_binding_sites(binding_model, 'MAX').max(1))
    
    def calc_bnd_frac(affinities, chem_pot):
        return calc_occ(chem_pot, affinities).mean()
    
    def f(u):
        bnd_frac = calc_bnd_frac(affinities, u)
        #print u, bnd_frac, prot_conc, prot_conc*bnd_frac, math.exp(u), \
        #    dna_conc*bnd_frac + math.exp(u)
        return prot_conc - math.exp(u) - dna_conc*bnd_frac
        #return prot_conc - math.exp(u) - prot_conc*bnd_frac

    min_u = -1000
    max_u = np.log(prot_conc/(R*T))
    rv = brentq(f, min_u, max_u, xtol=1e-4)
    return rv


def simulate_reads( binding_model, seq_len, sim_sizes,
                    dna_conc, prot_conc,
                    fname_prefix="test",
                    pool_size = 100000):
    current_pool = FixedLengthDNASequences(
        sample_random_seqs(pool_size, seq_len))
    chem_pots = []
    rnds_and_seqs = []
    current_pool = FixedLengthDNASequences(
        sample_random_seqs(pool_size, seq_len))
    for rnd, sim_size in enumerate(sim_sizes):
        # write the current pool sequences out to a file
        seq_indices = np.random.choice(
            len(current_pool), size=sim_size,
            replace=True)        
        seqs = FixedLengthDNASequences(
            (current_pool[i] for i in seq_indices))
        with open("%s_rnd_%i.fastq" % (fname_prefix, rnd), "w") as ofp:
            for i, seq in enumerate(seqs):
                ofp.write("@SIMSEQ%i\n%s\n+\n%s\n" % (i, seq, 'H'*len(seq)))
        
        # estimate the chemical potential
        chem_pot = est_chem_potential(
            current_pool, binding_model, dna_conc, prot_conc )
        chem_pots.append(chem_pot)

        # update the pool
        scores = current_pool.score_binding_sites(binding_model, 'MAX').max(1)
        occs = calc_occ(chem_pot, -scores)
        seq_indices = np.random.choice(
            len(current_pool), size=pool_size,
            p=occs/occs.sum(), replace=True)
        current_pool = FixedLengthDNASequences(
            (current_pool[i] for i in seq_indices))
        pyTFbindtools.log( 
            "Finished simulations for round %i" % rnd, level='VERBOSE')
    
    pyTFbindtools.log("Finished Simulations", level='VERBOSE')
    pyTFbindtools.log("Ref Energy: %.2f" % binding_model.ref_energy, level='VERBOSE')
    pyTFbindtools.log("Chem Pots: %s" % chem_pots, level='VERBOSE')
    pyTFbindtools.log(str(binding_model.ddg_array), level='VERBOSE')
    return

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Simulate a SELEX experiment.')

    parser.add_argument( '--tf-name', type=str,
                         help='A TF to simulate from.')
    
    parser.add_argument( '--sim-sizes', type=int, nargs='+',
                         help='Number of reads to simulate for each round.')    
    
    parser.add_argument( '--prot-conc', type=float, default=7.75e-10,
                         help='The protein concentration.  (mols/L)')
    parser.add_argument( '--dna-conc', type=float, default=2e-8,
                         help='The DNA concentration. (mols/L)')
    
    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e6,
        help='The random pool size for the bootstrap.')
    
    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    
    args = parser.parse_args()

    pyTFbindtools.VERBOSE = args.verbose
    
    if args.random_seed != None:
        np.random.seed(args.random_seed)

    mo = load_binding_models_from_db(tf_names=[args.tf_name,])[0]
    
    return ( mo, args.prot_conc, args.dna_conc, 
             args.sim_sizes,
             int(args.random_seq_pool_size) )

def main():
    ( binding_model, prot_conc, dna_conc, sim_sizes, random_seq_pool_size 
      ) = parse_arguments()

    simulate_reads( binding_model, 20, sim_sizes,
                    dna_conc, prot_conc,
                    pool_size = random_seq_pool_size)
    return

if __name__ == '__main__':
    main()
