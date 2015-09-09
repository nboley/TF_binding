import os, sys

from . import bootstrap_lhds, est_chem_potentials

n_dna_seq = 7.5e-8/(1.079734e-21*119) # molecules  - g/( g/oligo * oligo/molecule)
dna_conc = n_dna_seq/(6.02e23*5.0e-5) # mol/L
prot_conc = dna_conc/25 # mol/L (should be 25)

def main():
    motifs = load_motifs(sys.argv[1])
    motif = motifs.values()[0]

    ref_energy, ddg_array = motif.build_ddg_array()

    chem_pots = est_chem_potentials(
        ddg_array, ref_energy, 
        dna_conc, prot_conc, 
        n_bind_sites,
        len(sim_sizes))

    print ref_energy
    print chem_pots

main()
