import os, sys
import gzip
import random

from itertools import izip, chain
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_ind

import pyTFbindtools

import pyTFbindtools.selex

from pyTFbindtools.selex.log_lhd import calc_log_lhd, calc_binding_site_energies

from pyTFbindtools.selex import (
    find_pwm, code_seq, code_seqs, code_RC_seqs, calc_occ,
    estimate_dg_matrix_with_adadelta,
    find_pwm_from_starting_alignment,
    PartitionedAndCodedSeqs)
from pyTFbindtools.motif_tools import (
    build_pwm_from_energies,
    load_energy_data, load_motifs, load_motif_from_text,
    Motif, R, T,
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
jolma_dna_conc = n_dna_seq/(6.02e23*5.0e-5) # mol/L
jolma_prot_conc = jolma_dna_conc/25 # mol/L (should be 25)
#prot_conc *= 10

USE_FULL_BG_FOR_PART_FN = True
DEFAULT_MIN_NUM_BG_SEQS = 100000

def load_text_file(fp, maxnum=1e8):
    seqs = []
    for i, line in enumerate(fp):
        seqs.append(line.strip().upper())
        if i > maxnum: break
    return seqs

def load_fastq(fp, maxnum=1e8):
    seqs = []
    for i, line in enumerate(fp):
        if i/4 >= maxnum: break
        if i%4 == 1:
            seqs.append(line.strip().upper())
    return seqs

def load_sequences(fnames, max_num_seqs_per_file=1e8):
    fnames = list(fnames)
    rnds_and_seqs = {}
    rnd_nums = [int(x.split("_")[-1].split(".")[0]) for x in fnames]
    rnds_and_fnames = dict(zip(rnd_nums, fnames))
    for rnd, fname in rnds_and_fnames.iteritems():
        opener = gzip.open if fname.endswith(".gz") else open  
        with opener(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs[rnd] = loader(fp, max_num_seqs_per_file)
    return rnds_and_seqs

def load_sequence_data(seq_fps, 
                       background_seq_fp, 
                       max_num_seqs_per_file, 
                       min_num_background_sequences):
    pyTFbindtools.log("Loading sequences", 'VERBOSE')
    rnds_and_seqs = load_sequences(
        (x.name for x in seq_fps), max_num_seqs_per_file)

    if  max_num_seqs_per_file < min_num_background_sequences:
        pyTFbindtools.log(
            "WARNING: reducing the number of background sequences to --max-num-seqs-per-file ")
        min_num_background_sequences = max_num_seqs_per_file
    else:
        min_num_background_sequences = min_num_background_sequences
    
    background_seqs = None
    if background_seq_fp is not None:
        opener = ( gzip.open 
                   if background_seq_fp.name.endswith(".gz") 
                   else open  )
        with opener(background_seq_fp.name) as fp:
            background_seqs = load_fastq(fp, max_num_seqs_per_file)
    else:
        background_seqs = sample_random_seqs(
            min_num_background_sequences, 
            len(rnds_and_seqs.values()[0][0]))

    if len(background_seqs) < min_num_background_sequences:
        pyTFbindtools.log(
            "Too few (%i) background sequences were provided - sampling an additional %i uniform random sequences" % (
                len(background_seqs), 
                min_num_background_sequences-len(background_seqs)
            ), "VERBOSE")
        seq_len = len(rnds_and_seqs.values()[0][0])
        assert len(background_seqs) == 0 or len(background_seqs[0]) == seq_len,\
            "Background sequence length does not match sequence length."
        background_seqs.extend( 
            sample_random_seqs(
                min_num_background_sequences-len(background_seqs), seq_len)
        )

    return rnds_and_seqs, background_seqs

def initialize_starting_motif(
        pwm_fp, 
        energy_mo_fp, 
        rnds_and_seqs,
        initial_binding_site_len, 
        factor_name):
    assert (pwm_fp is None) or (energy_mo_fp is None), \
        "Cant initialize a motif from both a pwm and energy model"
    if pwm_fp is not None:
        pyTFbindtools.log("Loading PWM starting location", 'VERBOSE')
        motifs = load_motifs(pwm_fp)
        assert len(motifs) == 1, "Motif file contains multiple motifs"
        return motifs.values()[0]
    elif energy_mo_fp is not None:
        pyTFbindtools.log("Loading energy data", 'VERBOSE')
        return load_energy_data(energy_mo_fp.name)
    else:
        pyTFbindtools.log(
            "Initializing starting location from %imer search" % initial_binding_site_len, 
            'VERBOSE')
        bs_len = initial_binding_site_len
        pwm = find_pwm(rnds_and_seqs, initial_binding_site_len)
        motif = Motif("aligned_%imer" % initial_binding_site_len, 
                      factor_name, pwm)
        return motif
    assert False

def write_output(motif_name, ddg_array, ref_energy, ofp=sys.stdout):
    # normalize the array so that the consensus energy is zero
    consensus_energy = ddg_array.calc_min_energy(ref_energy)
    base_energies = ddg_array.calc_base_contributions()
    print >> ofp, ">%s.ENERGY\t%.6f" % (motif_name, consensus_energy)
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    conc_energies = []
    for pos, energies in enumerate(base_energies, start=1):
        conc_energies.append(
            energies - energies.min() - consensus_energy/len(base_energies))
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.6f" % (x - energies.min()) 
            for x in energies )

    print >> ofp, ">%s.PWM\t%s" % (motif_name, ddg_array.consensus_seq())
    pwm = build_pwm_from_energies(ddg_array, ref_energy, -12.0)
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    for pos, freqs in enumerate(pwm.T):
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.4f" % x for x in freqs )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate energy models from a SELEX experiment.')

    parser.add_argument( '--selex-files', nargs='+', type=file, required=True,
                         help='Files containing SELEX reads.')
    parser.add_argument( '--background-sequences', type=file, 
        help='File containing reads sequenced from round 0.')
    parser.add_argument( '--min-num-background-sequences', 
                         type=int, default=DEFAULT_MIN_NUM_BG_SEQS,
        help='Minimum number of background sequences (if less than %i are provided we simulate additional background sequences)' % DEFAULT_MIN_NUM_BG_SEQS
    )

    parser.add_argument( '--starting-pwm', type=file,
                         help='A PWM to start from.')
    parser.add_argument( '--starting-energy-model', type=file,
                         help='An energy model to start from.')
    parser.add_argument( '--initial-binding-site-len', type=int, default=6,
        help='The starting length of the binding site (this will grow)')

    parser.add_argument( '--random-seed', type=int,
                         help='Set the random number generator seed.')
    parser.add_argument( '--max-num-seqs-per-file', type=int, default=1e8,
                         help='Only load the first --max-num-seqs-per-file per input file (useful for debugging - overwrites --min-num-background-sequences)')
    parser.add_argument( '--max-iter', type=float, default=1e5,
                         help='Maximum number of optimization iterations.')
    parser.add_argument( '--partition-background-seqs', 
                         default=False, action='store_true',
        help='Use a subset of bnackground sequences to calculate the partition function (not recommended).')
    parser.add_argument( '--lhd-convergence-eps', type=float, default=1e-8,
                         help='Convergence tolerance for lhd change.')

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

    # parse the sequence data files
    rnds_and_seqs, background_seqs = load_sequence_data(
        args.selex_files, 
        args.background_sequences, 
        args.max_num_seqs_per_file, 
        args.min_num_background_sequences)
    # close the sequence files
    for fp in args.selex_files: fp.close()
    if args.background_sequences is not None: args.background_sequences.close()

    # load a starting motif, default to a search for the most over expressed 
    # sub-sequence if no starting model is provided
    motif = initialize_starting_motif(
        args.starting_pwm,
        args.starting_energy_model,
        rnds_and_seqs,
        args.initial_binding_site_len, 
        factor_name="TEST")
    # close the starting motif files
    if args.starting_pwm is not None: 
        args.starting_pwm.close()
    if args.starting_energy_model is not None: 
        args.starting_energy_model.close()

    return motif, rnds_and_seqs, background_seqs, args.partition_background_seqs

def sample_random_seqs(n_sims, seq_len):
    return ["".join(random.choice('ACGT') for j in xrange(seq_len))
            for i in xrange(n_sims)]

def find_best_shift(coded_seqs, ddg_array, coded_bg_seqs=None):
    def find_flanking_base_cnts(seqs, base_offset):
        # calculate the ddg energies for all binding sites. We use this to align
        # the binding sities within the sequences
        # find the index of the best offset
        energies = calc_binding_site_energies(seqs, ddg_array)
        best_offsets = np.argmin(energies, 1)

        ## find the binding sites which align to the sequence boundary. We must 
        ## remove these because the flanking sequence is not random, and so it
        ## will bias the results
        # deal with left shifts
        if base_offset > 0:
            base_offset += (ddg_array.motif_len - 1)
        base_indices = best_offsets + base_offset
        # find which offsets fit inside of the random portion of the sequence
        valid_indices = (base_indices > 0)&(base_indices < seq_len)
        base_cnts = seqs[
            valid_indices,:,base_indices[valid_indices]].sum(0)
        # add in the A counts
        base_cnts = np.insert(
            base_cnts, 0, valid_indices.sum()-base_cnts.sum())
        return base_cnts

    def calc_entropy(cnts):
        if len(cnts.shape) == 1: cnts = cnts[None,:]
        ps = np.array(cnts+1, dtype=float)/(cnts.sum(1)+4)[:,None]
        return -(ps*np.log(ps+1e-6)).sum(1)
    
    def estimate_entropy_significance(cnts, bg_cnts, n_samples=100000):
        bg_samples = np.random.multinomial(
            bg_cnts.sum(), (1.0+bg_cnts)/(bg_cnts.sum()+4.0), n_samples)
        obs_samples = np.random.multinomial(
            cnts.sum(), (1.0+cnts)/(cnts.sum()+4.0), n_samples)
        stat, p_value = ttest_ind(
            calc_entropy(bg_samples),
            calc_entropy(obs_samples)) 
        return stat, p_value

    seq_len = coded_seqs.shape[2]
    if coded_bg_seqs == None:
        coded_bg_seqs = sample_random_coded_seqs(coded_seqs.shape[0], seq_len)

    ### Code to downsample - I dont think that we want this but the 
    ### entropy calculation isn't perfect so Ill leave it
    #sample_size = min(coded_seqs.shape[0], coded_bg_seqs.shape[0])
    #coded_seqs = coded_seqs[
    #    np.random.choice(coded_seqs.shape[0], sample_size),:,:]
    #coded_bg_seqs = coded_bg_seqs[
    #    np.random.choice(coded_bg_seqs.shape[0], sample_size),:,:]
    
    entropies_and_offsets = []
    for offset in chain(xrange(max(-3, -seq_len+ddg_array.motif_len+1), 0), 
                        xrange(1,min(4, seq_len-ddg_array.motif_len))):
        cnts = find_flanking_base_cnts(coded_seqs, offset)
        bg_cnts = find_flanking_base_cnts(coded_bg_seqs, offset)
        t_stat, p_value = estimate_entropy_significance(cnts, bg_cnts)
        entropies_and_offsets.append(
            (-t_stat, 
             p_value, 
             offset, 
             calc_entropy(cnts)[0], 
             calc_entropy(bg_cnts)[0]))
    entropies_and_offsets.sort()

    for x in entropies_and_offsets:
        print x
    print
    
    if len(entropies_and_offsets) == 0:
        return None

    if entropies_and_offsets[0][0] > 0 or entropies_and_offsets[0][1] > 0.05:
        return None
    
    if entropies_and_offsets[0][2] > 0:
        return 'RIGHT'
    return 'LEFT'

def fit_model(rnds_and_seqs, background_seqs, 
              ddg_array, ref_energy, 
              dna_conc, prot_conc,
              partition_background_seqs):
    pyTFbindtools.log("Coding sequences", 'VERBOSE')
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs, 
        background_seqs, 
        use_full_background_for_part_fn=(not partition_background_seqs)
    )

    #
    #partitioned_and_coded_bg_seqs = PartitionedAndCodedSeqs(
    #    {0: background_seqs}, 
    #    ( 1 if USE_FULL_BG_FOR_PART_FN else 
    #      partitioned_and_coded_rnds_and_seqs.n_partitions )
    #)
    
    opt_path = []
    prev_lhd = None
    while True:
        bs_len = ddg_array.motif_len
        prev_lhd = calc_log_lhd(
            ref_energy, ddg_array, 
            partitioned_and_coded_rnds_and_seqs.validation.bnd_seqs,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs,
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
            partitioned_and_coded_rnds_and_seqs.validation.bnd_seqs,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs,
            dna_conc,
            prot_conc)

        opt_path.append([bs_len, new_lhd, ddg_array, ref_energy])

        pyTFbindtools.log("Prev: %.2f\tCurr: %.2f\tDiff: %.2f" % (
            prev_lhd, new_lhd, new_lhd-prev_lhd), 'VERBOSE')

        with open("FITMO.BSLEN%i.txt" % bs_len, "w") as ofp:
            write_output("UNKNOWN", ddg_array, ref_energy, ofp)
        
        if ( bs_len >= 20 
             or bs_len+1 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break
        
        pyTFbindtools.log("Finding best shift", 'VERBOSE')
        shift_type = find_best_shift(
            partitioned_and_coded_rnds_and_seqs.validation.last_rnd,
            ddg_array,
            partitioned_and_coded_rnds_and_seqs.validation.bg_seqs)
        if shift_type == None:
            pyTFbindtools.log("Model has finished fitting", 'VERBOSE')
            break
            
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
    (motif, rnds_and_seqs, background_seqs, partition_background_seqs
     ) = parse_arguments()
    ref_energy, ddg_array = motif.build_ddg_array()
    ddg_array_hat, ref_energy_hat = fit_model(
        rnds_and_seqs, background_seqs, 
        ddg_array, ref_energy,
        jolma_dna_conc, jolma_prot_conc,
        partition_background_seqs
    )
    
    with open(motif.name + ".SELEX.txt", "w") as ofp:
        write_output(motif, ddg_array_hat, ref_energy_hat, ofp)
    
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
