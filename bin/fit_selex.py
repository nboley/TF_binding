import os, sys
import gzip

from itertools import izip
from collections import defaultdict

sys.path.insert(0, "/home/nboley/src/TF_binding/")

import numpy as np

import pyTFbindtools

import pyTFbindtools.selex

from pyTFbindtools.selex.log_lhd import calc_log_lhd, calc_binding_site_energies

from pyTFbindtools.selex import (
    find_pwm, code_seq, code_seqs, code_RC_seqs, calc_occ,
    estimate_dg_matrix_with_adadelta,
    find_pwm_from_starting_alignment,
    PartitionedAndCodedSeqs, base_map,
    sample_random_coded_seqs)
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
#prot_conc *= 10

def load_text_file(fp, maxnum=1e9):
    seqs = []
    for i, line in enumerate(fp):
        seqs.append(line.strip().upper())
        if i > maxnum: break
    return seqs

def load_fastq(fp, maxnum=1e9):
    seqs = []
    for i, line in enumerate(fp):
        if i/4 >= maxnum: break
        if i%4 == 1:
            seqs.append(line.strip().upper())
    return seqs

def load_sequences(fnames):
    fnames = list(fnames)
    rnds_and_seqs = {}
    rnd_nums = [int(x.split("_")[-1].split(".")[0]) for x in fnames]
    rnds_and_fnames = dict(zip(rnd_nums, fnames))
    for rnd, fname in rnds_and_fnames.iteritems():
        opener = gzip.open if fname.endswith(".gz") else open  
        with opener(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs[rnd] = loader(fp) # , 178
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

    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--random-seq-pool-size', type=float, default=1e6,
        help='The random pool size for the bootstrap.')


    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    parser.add_argument( '--debug-verbose', default=False, action='store_true',
                         help='Print debug information.')
    
    args = parser.parse_args()
    assert not (args.starting_pwm and args.starting_energy_model), \
            "Can not set both --starting-pwm and --starting-energy_model"

    pyTFbindtools.VERBOSE = args.verbose or args.debug_verbose
    pyTFbindtools.DEBUG = args.debug_verbose

    pyTFbindtools.selex.CONVERGENCE_MAX_LHD_CHANGE = args.lhd_convergence_eps
    pyTFbindtools.selex.MAX_NUM_ITER = int(args.max_iter)
    
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
    pwm = np.zeros((4, ddg_array.motif_len), dtype=float)
    mean_energy = ref_energy + chem_pot + ddg_array.mean_energy
    for i, base_energies in enumerate(ddg_array.calc_base_contributions()):
        base_mut_energies = mean_energy + base_energies.mean() - base_energies 
        occs = logistic(base_mut_energies)
        pwm[:,i] = occs/occs.sum()
    return pwm

def calc_entropy(cnts):
    ps = np.array(cnts, dtype=float)/cnts.sum()
    return -(ps*np.log(ps+1e-6)).sum()

def find_best_shift(coded_seqs, ddg_array):
    # calculate the ddg energies for all binding sites. We use this to align
    # the binding sities within the sequences
    energies = calc_binding_site_energies(coded_seqs, ddg_array)
    # find the index of the best offset
    best_offsets = np.argmin(energies, 1)
    # find the binding sites which align to the sequence boundary. We must 
    # remove these because the flanking sequence is not random, and so it
    # will bias the results
    max_offset = coded_seqs.shape[2] - ddg_array.motif_len
    bndry_alignments = (best_offsets > 0)&(best_offsets < max_offset)
    non_bndry_best_offsets = best_offsets[bndry_alignments]

    # find the distribution of bases left of the aligned binding sites
    left_base_cnts = coded_seqs[
        bndry_alignments,:,non_bndry_best_offsets-1].sum(0)
    left_base_cnts = np.insert(
        left_base_cnts, 0, bndry_alignments.shape[0]-left_base_cnts.sum())
    left_base_score = calc_entropy(left_base_cnts)

    # find the distribution of bases to the right of the aligned binding sites
    right_base_cnts = coded_seqs[
        bndry_alignments,:,non_bndry_best_offsets+ddg_array.motif_len].sum(0)
    right_base_cnts = np.append(
        right_base_cnts, bndry_alignments.shape[0]-right_base_cnts.sum())
    right_base_score = calc_entropy(right_base_cnts)
    
    if left_base_score < right_base_score:
        return 'LEFT'
    return 'RIGHT'

def test_RC_equiv():
    """Make sure that the RC function works correctly.
    
    """
    seq = 'GCGAATACC'
    coded_seq = code_seq(seq)
    RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    RC_seq = "".join(RC_map[x] for x in seq[::-1])
    coded_RC_seq = code_seq(RC_seq)
    
    print coded_seq
    print coded_RC_seq
    print calc_binding_site_energies(coded_seq[None,(1,2,3),:], ddg_array)
    print calc_binding_site_energies(coded_RC_seq[None, (1,2,3),:], ddg_array)[:,::-1]
    energy_diff, RC_ddg_array = ddg_array.reverse_complement()
    print energy_diff + calc_binding_site_energies(coded_seq[None, (1,2,3),:], RC_ddg_array)
    #print -energy_diff + calc_binding_site_energies(coded_seq[None,(1,2,3),:], RC_ddg_array)
    return

    pass

def fit_model(rnds_and_seqs, ddg_array, ref_energy):
    pyTFbindtools.log("Coding sequences", 'VERBOSE')
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs)

    opt_path = []
    prev_lhd = None
    while True:
        bs_len = ddg_array.motif_len
        prev_lhd = calc_log_lhd(
            ref_energy, ddg_array, 
            partitioned_and_coded_rnds_and_seqs[0], 
            dna_conc, prot_conc)
        pyTFbindtools.log("Starting lhd: %.2f" % prev_lhd, 'VERBOSE')
        
        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, lhd_path, lhd_hat 
            ) = estimate_dg_matrix_with_adadelta(
                partitioned_and_coded_rnds_and_seqs,
                ddg_array, ref_energy,
                dna_conc, prot_conc)

        pyTFbindtools.log(ddg_array.consensus_seq(), 'VERBOSE')
        pyTFbindtools.log("Ref: %s" % ref_energy, 'VERBOSE')
        pyTFbindtools.log(
            "Mean: %s" % (ref_energy + ddg_array.sum()/3), 'VERBOSE')
        pyTFbindtools.log(
            "Min: %s" % ddg_array.calc_min_energy(ref_energy), 'VERBOSE')
        pyTFbindtools.log(
            str(ddg_array.calc_base_contributions().round(2)), 'VERBOSE')

        new_lhd = calc_log_lhd(
            ref_energy, 
            ddg_array, 
            partitioned_and_coded_rnds_and_seqs[0],
            dna_conc,
            prot_conc)

        opt_path.append([bs_len, new_lhd, ddg_array, ref_energy])

        pyTFbindtools.log("Prev: %.2f\tCurr: %.2f\tDiff: %.2f" % (
            prev_lhd, new_lhd, new_lhd-prev_lhd), 'VERBOSE')

        if prev_lhd + 10 > lhd_hat:
            pyTFbindtools.log("Model has finished fitting", 'VERBOSE')
            break
        
        if ( bs_len >= 20 
             or bs_len+1 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break
        
        pyTFbindtools.log("Finding best shift", 'VERBOSE')
        shift_type = find_best_shift(
            partitioned_and_coded_rnds_and_seqs.validation[max(
                partitioned_and_coded_rnds_and_seqs.validation.keys())],
            ddg_array)
        if shift_type == 'LEFT':
            pyTFbindtools.log("Adding left base to motif", level='VERBOSE' )
            ddg_array = np.hstack((np.zeros((3,1), dtype='float32'), ddg_array)
                              ).view(DeltaDeltaGArray)
        elif shift_type == 'RIGHT':
            pyTFbindtools.log("Adding right base to motif", level='VERBOSE' )
            ddg_array = np.hstack((ddg_array, np.zeros((3,1), dtype='float32'))).view(
                DeltaDeltaGArray)
        else:
            assert False, "Unrecognized shift type '%s'" % shift_type
        ref_energy = ref_energy
        
    for entry in opt_path:
        print entry
    
    return ddg_array, ref_energy
    
def main():
    motif, rnds_and_seqs, random_seq_pool_size = parse_arguments()
    ref_energy, ddg_array = motif.build_ddg_array()
    ddg_array_hat, ref_energy_hat = fit_model(
        rnds_and_seqs, ddg_array, ref_energy )
    
    with open(motif.name + ".SELEX.txt", "w") as ofp:
        write_output(motif, ddg_array_hat, ref_energy_hat, ofp)
    
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
