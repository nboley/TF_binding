import os, sys

import numpy as np

sys.path.insert(0, "/home/nboley/src/TF_binding/")

import pyTFbindtools
#from pyTFbindtools.selex import est_chem_potentials
from pyTFbindtools.motif_tools import load_energy_data, load_motifs

from pyDNAbinding.DB import load_binding_models_from_db
from pyDNAbinding.sequence import sample_random_seqs
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ

def simulate_reads( binding_model, seq_len, sim_sizes,
                    dna_conc, prot_conc,
                    fname_prefix="test",
                    pool_size = 100000):
    #chem_pots = est_chem_potentials(
    #    ddg_array, ref_energy, dna_conc, prot_conc, 
    #    2*(seq_len-len(motif)+1), len(sim_sizes))
    chem_pots = [-8.0]*len(sim_sizes)
    rnds_and_seqs = []
    current_pool = FixedLengthDNASequences(
        sample_random_seqs(pool_size, seq_len))
    for rnd, (sim_size, chem_pot) in enumerate(
            zip(sim_sizes, chem_pots), start=0):
        #print current_pool
        seq_indices = np.random.choice(
            len(current_pool), size=sim_size,
            replace=True)        
        seqs = FixedLengthDNASequences(
            (current_pool[i] for i in seq_indices))
        with open("%s_rnd_%i.fastq" % (fname_prefix, rnd), "w") as ofp:
            for i, seq in enumerate(seqs):
                ofp.write("@SIMSEQ%i\n%s\n+\n%s\n" % (i, seq, 'H'*len(seq)))

        # update the pool
        scores = current_pool.score_binding_sites(binding_model, 'MAX').max(1)
        occs = calc_occ(chem_pot, -scores)
        seq_indices = np.random.choice(
            len(current_pool), size=pool_size,
            p=occs/occs.sum(), replace=True)
        current_pool = FixedLengthDNASequences((seqs[i] for i in seq_indices))
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
                         help='The protein concentration.')
    parser.add_argument( '--dna-conc', type=float, default=2e-8,
                         help='The DNA concentration.')
    
    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e2,
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
