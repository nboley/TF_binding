import os, sys
import gzip

from itertools import izip

sys.path.insert(0, "/home/nboley/src/TF_binding/")

import numpy as np

import pyTFbindtools

import pyTFbindtools.selex

from pyTFbindtools.selex import (
    find_pwm, code_seqs, estimate_dg_matrix,
    estimate_dg_matrix_with_adadelta,
    est_chem_potentials, bootstrap_lhds,
    find_pwm_from_starting_alignment,
    partition_and_code_all_seqs, calc_log_lhd_factory)
from pyTFbindtools.motif_tools import (
    load_energy_data, load_motifs, load_motif_from_text,
    logistic, Motif, R, T,
    DeltaDeltaGArray)

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
# DNA is on average ~ 1.079734e-21 g/oligo * 119 oligos
# molar protein:DNA ratio: 1:25
# volume: 5.0e-5 L 
n_dna_seq = 7.5e-8/(1.079734e-21*119) # molecules  - g/( g/oligo * oligo/molecule)
dna_conc = n_dna_seq/(6.02e23*5.0e-5) # mol/L
prot_conc = dna_conc/25 # mol/L (should be 25)
#prot_conc /= 1000

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

def load_sequences(fnames):
    rnds_and_seqs = []
    for fname in sorted(fnames,
                        key=lambda x: int(x.split("_")[-1].split(".")[0])):
        opener = gzip.open if fname.endswith(".gz") else open  
        with opener(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs.append( loader(fp) ) # [:1000]
    return rnds_and_seqs

def write_output(motif, ddg_array, ref_energy, ofp=sys.stdout):
    # normalize the array so that the consensus energy is zero
    consensus_energy = ddg_array.calc_min_energy(ref_energy)
    base_energies = ddg_array.calc_base_contributions()
    print >> ofp, ">%s.ENERGY\t%.6f" % (motif.name, consensus_energy)
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    conc_energies = []
    for pos, energies in enumerate(base_energies, start=1):
        conc_energies.append(
            energies - energies.min() - consensus_energy/len(base_energies))
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.6f" % (x - energies.min()) 
            for x in energies )

    print >> ofp, ">%s.PWM" % motif.name
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    for pos, energies in enumerate(conc_energies, start=1):
        pwm = 1-logistic(energies)
        pwm = pwm/pwm.sum()
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.4f" % x for x in pwm )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate energy models from a SELEX experiment.')

    parser.add_argument( '--selex-files', nargs='+', type=file, required=True,
                         help='Files containing SELEX reads.')

    parser.add_argument( '--background-sequence', type=file, 
        help='File containing reads sequenced from round 0.')

    parser.add_argument( '--starting-pwm', type=file,
                         help='A PWM to start from.')
    parser.add_argument( '--starting-energy-model', type=file,
                         help='An energy model to start from.')
    parser.add_argument( '--initial-binding-site-len', type=int, default=6,
        help='The starting length of the binding site (this will grow)')

    parser.add_argument( '--lhd-convergence-eps', type=float, default=1e-8,
                         help='Convergence tolerance for lhd change.')
    parser.add_argument( '--max-iter', type=float, default=1e5,
                         help='Maximum number of optimization iterations.')
    parser.add_argument( '--momentum', type=float, default=0.1,
                         help='Optimization tuning param (between 0 and 1).')

    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e5,
        help='The random pool size for the bootstrap.')


    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    
    args = parser.parse_args()
    assert not (args.starting_pwm and args.starting_energy_model), \
            "Can not set both --starting-pwm and --starting-energy_model"

    pyTFbindtools.VERBOSE = args.verbose
    pyTFbindtools.selex.CONVERGENCE_MAX_LHD_CHANGE = args.lhd_convergence_eps
    pyTFbindtools.selex.MAX_NUM_ITER = int(args.max_iter)
    assert args.momentum < 1 and args.momentum >= 0
    pyTFbindtools.selex.MOMENTUM = args.momentum
    
    if args.random_seed != None:
        np.random.seed(args.random_seed)

    pyTFbindtools.log("Loading sequences", 'VERBOSE')
    rnds_and_seqs = load_sequences(x.name for x in args.selex_files)

    if args.starting_pwm != None:
        pyTFbindtools.log("Loading PWM starting location", 'VERBOSE')
        motifs = load_motifs(args.starting_pwm)
        assert len(motifs) == 1, "Motif file contains multiple motifs"
        motif = motifs.values()[0]
        args.starting_pwm.close()
    elif args.starting_energy_model != None:
        pyTFbindtools.log("Loading energy data", 'VERBOSE')
        motif = load_energy_data(args.starting_energy_model.name)
        args.starting_energy_model.close()
    else:
        pyTFbindtools.log(
            "Initializing starting location from %imer search" % args.initial_binding_site_len, 
            'VERBOSE')
        factor_name = 'TEST'
        bs_len = args.initial_binding_site_len
        pwm = find_pwm(rnds_and_seqs, args.initial_binding_site_len)
        motif = Motif("aligned_%imer" % args.initial_binding_site_len, 
                      factor_name, pwm)
    
    return motif, rnds_and_seqs, int(args.random_seq_pool_size)

def build_pwm_from_energies(ddg_array, ref_energy, chem_pot):
    from pyTFbindtools.selex import build_random_read_energies_pool, calc_occ
    pwm = np.zeros((len(ddg_array)/3, 4))
    energies, seqs = build_random_read_energies_pool(
        10000, len(ddg_array)/3, ddg_array, ref_energy, store_seqs=True)
    occs = calc_occ(energies, ref_energy, chem_pot)
    for seq, occ in izip(seqs, occs):
        for base_pos, base in enumerate(seq):
            pwm[base_pos,base] += occ
    pwm = pwm.T/pwm.sum(1)
    return pwm

def find_best_shift(rnds_and_seqs, ddg_array, ref_energy):
    pwm = find_pwm_from_starting_alignment(
        rnds_and_seqs[-1], build_pwm_from_energies(ddg_array, ref_energy, -12))
    left_shift_pwm = find_pwm_from_starting_alignment(
        rnds_and_seqs[-1], np.hstack((np.zeros((4,1)), pwm.T)))    
    right_shift_pwm = find_pwm_from_starting_alignment(
        rnds_and_seqs[-1], np.hstack((pwm.T, np.zeros((4,1)))))    
    left_shift_score = (
        left_shift_pwm[0,:]*np.log(left_shift_pwm[0,:])).sum()
    right_shift_score = (
        right_shift_pwm[-1,:]*np.log(right_shift_pwm[-1,:])).sum()
    if left_shift_score < right_shift_score:
        return "LEFT"
    else:
        return "RIGHT"

def fit_model(rnds_and_seqs, ddg_array, ref_energy, random_seq_pool_size):
    opt_path = []
    prev_shift_type = None
    for rnd_num in xrange(len(rnds_and_seqs[0][0])-ddg_array.motif_len+1):
        bs_len = ddg_array.motif_len
        
        pyTFbindtools.log("Coding sequences", 'VERBOSE')
        partitioned_and_coded_rnds_and_seqs = partition_and_code_all_seqs(
            rnds_and_seqs, bs_len)

        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, lhd_hat
        ) = estimate_dg_matrix_with_adadelta(
            partitioned_and_coded_rnds_and_seqs,
            ddg_array, ref_energy,
            dna_conc, prot_conc)

        if prev_shift_type != None:
            pyTFbindtools.log("checking quality of fit", 'VERBOSE')
            calc_log_lhd = calc_log_lhd_factory(partitioned_and_coded_rnds_and_seqs)
            n_bind_sites = partitioned_and_coded_rnds_and_seqs[0][0].get_value().shape[1]
            chem_pots = est_chem_potentials(
                ddg_array, ref_energy, 
                dna_conc, prot_conc, 
                n_bind_sites,
                len(partitioned_and_coded_rnds_and_seqs[0]))
            alt_lhd = calc_log_lhd(ref_energy, ddg_array, chem_pots, 0)
            train_lhds = [
                calc_log_lhd(ref_energy, ddg_array, chem_pots, i)
                for i in xrange(1, len(partitioned_and_coded_rnds_and_seqs)) ]
            null_ddg_array = ddg_array.copy()
            if prev_shift_type == 'LEFT':
                null_ddg_array[:3] = 0
            elif prev_shift_type == 'RIGHT':
                null_ddg_array[-3:] = 0
            chem_pots = est_chem_potentials(
                ddg_array, ref_energy, 
                dna_conc, prot_conc, 
                n_bind_sites,
                len(partitioned_and_coded_rnds_and_seqs[0]))
            null_lhd = calc_log_lhd(ref_energy, null_ddg_array, chem_pots, 0)
            print "Alt", alt_lhd, "Null", null_lhd
            print "Alt SD:", np.std(np.array(train_lhds))
            print "Alt Mean:", np.mean(np.array(train_lhds))
            print train_lhds
            # if the lhd change wasn't sufficeintly big, we are done
            if null_lhd - alt_lhd < 10:
                break
            
        shift_type = find_best_shift(rnds_and_seqs, ddg_array, ref_energy)
        if shift_type == 'LEFT':
            pyTFbindtools.log("Adding left base to motif", level='VERBOSE' )
            ddg_array = np.insert(ddg_array, 0, np.zeros(3, dtype='float32')
                              ).view(DeltaDeltaGArray)
        elif shift_type == 'RIGHT':
            pyTFbindtools.log("Adding right base to motif", level='VERBOSE' )
            ddg_array = np.append(ddg_array, np.zeros(3, dtype='float32')).view(
                DeltaDeltaGArray)
        else:
            assert False, "Unrecognized shift type '%s'" % shift_type
        ref_energy = ref_energy
        prev_shift_type = shift_type 
        for i in xrange(10):
            print "="*100
   
    return ddg_array, ref_energy
    
def main():
    motif, rnds_and_seqs, random_seq_pool_size = parse_arguments()
    ref_energy, ddg_array = motif.build_ddg_array()
    #pwm = find_pwm_from_starting_alignment(
    #    rnds_and_seqs, ddg_array.base_contributions())
    #print pwm
    #print build_pwm_from_energies(ddg_array, ref_energy, -12)
    #assert False
    #from pyTFbindtools.selex import est_partition_fn
    #est_partition_fn(ref_energy, ddg_array, 2)
    #return
    ddg_array_hat, ref_energy_hat = fit_model(
        rnds_and_seqs, ddg_array, ref_energy,
        random_seq_pool_size)
    
    with open(motif.name + ".SELEX.txt", "w") as ofp:
        write_output(motif, ddg_array_hat, ref_energy_hat, ofp)
    
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
