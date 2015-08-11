import os, sys

import numpy as np

sys.path.insert(0, "/home/nboley/src/TF_binding/")

import pyTFbindtools
from pyTFbindtools.selex import est_chem_potentials
from pyTFbindtools.motif_tools import load_energy_data, load_motifs


def simulate_reads( motif, seq_len, sim_sizes,
                    dna_conc, prot_conc,
                    fname_prefix="test",
                    pool_size = 100000):
    ref_energy, ddg_array = motif.build_ddg_array()
    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, dna_conc, prot_conc, 
        2*(seq_len-len(motif)+1), len(sim_sizes))
    current_pool = np.array([np.random.randint(4, size=seq_len)
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

        with open("%s_rnd_%i.txt" % (fname_prefix, rnd), "w") as ofp:
            for seq in seqs:
                print >> ofp, "".join('ACGT'[x] for x in seq)
        current_pool = seqs[np.random.choice(
            len(seqs), size=pool_size,
            p=seq_occs/seq_occs.sum(), replace=True)]
        pyTFbindtools.log( 
            "Finished simulations for round %i" % rnd, level='VERBOSE')
    
    pyTFbindtools.log("Finished Simulations", level='VERBOSE')
    pyTFbindtools.log("Ref Energy: %.2f" % ref_energy, level='VERBOSE')
    pyTFbindtools.log("Chem Pots: %s" % chem_pots, level='VERBOSE')
    pyTFbindtools.log(str(ddg_array), level='VERBOSE')
    return

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Simulate a SELEX experiment.')

    parser.add_argument( '--energy-model', type=file,
                         help='An energy model to simulate from.')
    parser.add_argument( '--pwm', type=file,
                         help='A pwm to simulate from.')
    
    parser.add_argument( '--sim-sizes', type=int, nargs='+',
                         help='Number of reads to simulate for each round.')    
    
    parser.add_argument( '--prot-conc', type=float, default=7.75e-10,
                         help='The protein concentration.')
    parser.add_argument( '--dna-conc', type=float, default=2e-8,
                         help='The DNA concentration.')
    
    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e5,
        help='The random pool size for the bootstrap.')
    
    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    
    args = parser.parse_args()

    pyTFbindtools.VERBOSE = args.verbose
    
    if args.random_seed != None:
        np.random.seed(args.random_seed)

    if args.pwm != None:
        pyTFbindtools.log("Loading PWM starting location", 'VERBOSE')
        motifs = load_motifs(args.pwm.name)
        assert len(motifs) == 1, "Motif file contains multiple motifs"
        motif = motifs.values()[0][0]
        args.pwm.close()
    else:
        assert args.energy_model != None, \
            "Either --energy-model or --pwm must be specified"
        pyTFbindtools.log("Loading energy data", 'VERBOSE')
        motif = load_energy_data(args.energy_model.name)
        args.starting_energy_model.close()
    
    return ( motif, args.prot_conc, args.dna_conc, 
             args.sim_sizes,
             int(args.random_seq_pool_size) )

def main():
    ( motif, prot_conc, dna_conc, sim_sizes, random_seq_pool_size 
      ) = parse_arguments()

    simulate_reads( motif, 20, sim_sizes,
                    dna_conc, prot_conc,
                    pool_size = random_seq_pool_size)
    return

if __name__ == '__main__':
    main()
