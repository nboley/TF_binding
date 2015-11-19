import os, sys
import gzip
import random

from itertools import izip, chain
from collections import defaultdict, namedtuple

import numpy as np

from pyDNAbinding.binding_model import EnergeticDNABindingModel, PWMBindingModel

import pyTFbindtools

import pyTFbindtools.selex

from pyTFbindtools.selex import (
    PartitionedAndCodedSeqs, ReducedDeltaDeltaGArray,
    progressively_fit_model, find_pwm, sample_random_seqs )

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

def optional_gzip_open(fname):
    return gzip.open(fname) if fname.endswith(".gz") else open(fname)  

def get_experiment_to_process_from_DB_queue(host, dbname, user):
    """XXX Untested

    """
    raise NotImplementedError, "This function is old and untested"
    conn = psycopg2.connect(
        "host=%s dbname=%s user=%s" % (host, dbname, user))
    cur = conn.cursor()
    query = """
    LOCK TABLE pending_selex_experiment IN ACCESS EXCLUSIVE MODE;
    SELECT * FROM pending_selex_experiment LIMIT 1;
    """
    cur.execute(query)
    exp_ids = [x[0] for x in cur.fetchall()]
    if len(exp_ids) == 0:
        conn.commit()
        return None
    else:
        exp_id = int(exp_ids[0])
        query = "DELETE FROM pending_selex_experiment WHERE selex_exp_id = '%s'"
        cur.execute(query, [exp_id,])
        conn.commit()
        return SelexDBConn(host, dbname, user, exp_id)


class SelexDBConn(object):
    def __init__(self, host, dbname, user, exp_id):
        import psycopg2
        self.conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % (host, dbname, user))
        self.exp_id = exp_id
        return

    def insert_model_into_db(
            self, ref_energy, ddg_array, validation_lhd):
        motif_len = ddg_array.motif_len
        consensus_energy, base_contributions = ddg_array.calc_normalized_base_conts(ref_energy)
        cur = self.conn.cursor()    
        query = """
        INSERT INTO new_selex_models
          (selex_exp_id, motif_len, consensus_energy, ddg_array, validation_lhd)
          VALUES 
          (%s, %s, %s, %s, %s) 
        RETURNING key
        """
        cur.execute(query, (
            self.exp_id, 
            motif_len, 
            float(consensus_energy), 
            base_contributions.tolist(),
            float(validation_lhd)
        ))
        self.conn.commit()
        return

    def get_fnames(self):
        cur = self.conn.cursor()
        query = """
        SELECT rnd, primer, fname 
          FROM selex_round
         WHERE selex_exp_id = '%i'
         ORDER BY rnd;
        """
        cur.execute(query % self.exp_id)
        primers = set()
        fnames = {}
        for rnd, primer, fname in cur.fetchall():
            fnames[int(rnd)] = fname
            primers.add(primer)
        
        assert len(primers) == 1
        primer = list(primers)[0]
        query = """
        SELECT fname 
          FROM selex_background_sequences
         WHERE primer = '%s';
        """
        cur.execute(query % primer)
        res = cur.fetchall()
        assert len(res) == 1
        bg_fname = res[0][0]
        
        return fnames, bg_fname

    def get_dna_and_prot_conc(self):
        cur = self.conn.cursor()
        query = """
        SELECT dna_conc, prot_conc 
          FROM selex_round
         WHERE selex_exp_id = '%i';
        """
        cur.execute(query % self.exp_id)
        concs = set(cur.fetchall())
        assert len(concs) == 1
        return concs.pop()

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
        with optional_gzip_open(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs[rnd] = loader(fp, max_num_seqs_per_file)
    return rnds_and_seqs

def load_sequence_data(selex_db_conn,
                       seq_fps,
                       background_seq_fp, 
                       max_num_seqs_per_file, 
                       min_num_background_sequences):
    if selex_db_conn is not None:
        assert seq_fps is None and background_seq_fp is None
        fnames, bg_fname = selex_db_conn.get_fnames()
        seq_fps = [ optional_gzip_open(fname) for fname in fnames.values() ]
        background_seq_fp = optional_gzip_open(bg_fname)

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
        with optional_gzip_open(background_seq_fp.name) as fp:
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
        assert False, "NOT IMPLEMENTED"
        pyTFbindtools.log("Loading PWM starting location", 'VERBOSE')
        motifs = load_motifs(pwm_fp)
        assert len(motifs) == 1, "Motif file contains multiple motifs"
        return motifs.values()[0]
    elif energy_mo_fp is not None:
        assert False, "NOT IMPLEMENTED"
        pyTFbindtools.log("Loading energy data", 'VERBOSE')
        return load_energy_data(energy_mo_fp.name)
    else:
        pyTFbindtools.log(
            "Initializing starting location from %imer search" % initial_binding_site_len, 
            'VERBOSE')
        bs_len = initial_binding_site_len
        pwm = find_pwm(rnds_and_seqs, initial_binding_site_len)
        pwm_model = PWMBindingModel(pwm, tf_name=factor_name)
        return pwm_model.build_energetic_model(include_shape=True)
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
    pwm = build_pwm(ddg_array, ref_energy, -12.0)
    #print >> ofp, "\t".join(["pos", "A", "C", "G", "T"])
    for pos, freqs in enumerate(pwm.T):
        print >> ofp, str(pos) + "\t" + "\t".join(
            "%.4f" % x for x in freqs )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate energy models from a SELEX experiment.')

    parser.add_argument( '--experiment-id', type=int,
        help='Use SELEX files associated with this experiment ID and put fit models into the DB.')
    
    parser.add_argument( '--selex-files', nargs='+', type=file,
                         help='Files containing SELEX reads.')
    parser.add_argument( '--background-sequences', type=file, 
        help='File containing reads sequenced from round 0.')
    parser.add_argument( '--ofname-prefix', type=str, 
        help='Output filename prefix (if not set do not write output to file)')

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

    parser.add_argument( '--min-num-background-sequences', 
                         type=int, default=DEFAULT_MIN_NUM_BG_SEQS,
        help='Minimum number of background sequences (if less than %i are provided we simulate additional background sequences)' % DEFAULT_MIN_NUM_BG_SEQS
    )
    parser.add_argument( '--partition-background-seqs', 
                         default=False, action='store_true',
        help='Use a subset of background sequences to calculate the partition function (not recommended).')

    parser.add_argument( '--verbose', default=False, action='store_true',
                         help='Print extra status information.')
    parser.add_argument( '--debug-verbose', default=False, action='store_true',
                         help='Print debug information.')
    
    args = parser.parse_args()
    assert not (args.starting_pwm is not None and args.starting_energy_model is not None), \
            "Can not set both --starting-pwm and --starting-energy_model"
    
    assert not (args.experiment_id is not None and args.selex_files is not None), \
            "Can not set both --experiment-id and --selex-files"    
    assert not (args.experiment_id is not None and args.background_sequences is not None), \
            "Can not set both --experiment-id and --background-sequences"    

    assert (args.experiment_id is not None or args.ofname_prefix is not None), \
        "Must either set an experiment id to write results to the DB, or specify an output filename prefix"
    
    pyTFbindtools.VERBOSE = args.verbose or args.debug_verbose
    pyTFbindtools.DEBUG = args.debug_verbose

    pyTFbindtools.selex.MAX_NUM_ITER = int(args.max_iter)
        
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    if args.experiment_id is not None:
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', args.experiment_id)
    
    # parse the sequence data files
    rnds_and_seqs, background_seqs = load_sequence_data(
        selex_db_conn,
        args.selex_files, 
        args.background_sequences, 
        args.max_num_seqs_per_file, 
        args.min_num_background_sequences)

    # close the sequence files
    if args.selex_files is not None:
        for fp in args.selex_files: fp.close()
    if args.background_sequences is not None: 
        args.background_sequences.close()

    # load a starting motif, default to a search for the most over expressed 
    # sub-sequence if no starting model is provided
    motif = initialize_starting_motif(
        args.starting_pwm,
        args.starting_energy_model,
        rnds_and_seqs,
        args.initial_binding_site_len, 
        factor_name=args.ofname_prefix)
    # close the starting motif files
    if args.starting_pwm is not None: 
        args.starting_pwm.close()
    if args.starting_energy_model is not None: 
        args.starting_energy_model.close()

    return ( motif, 
             rnds_and_seqs, background_seqs, 
             selex_db_conn, 
             args.partition_background_seqs )

def fit_model(rnds_and_seqs, background_seqs, 
              initial_model, 
              dna_conc, prot_conc,
              partition_background_seqs,
              selex_db_conn=None,
              output_fname_prefix=None):
    assert selex_db_conn is not None or output_fname_prefix is not None

    ref_energy, ddg_array = initial_model.ref_energy, initial_model.ddg_array
    assert (ddg_array[:,0].round(6) == 0).all()
    ddg_array = ddg_array[:,1:].T.view(ReducedDeltaDeltaGArray)
    
    pyTFbindtools.log("Coding sequences", 'VERBOSE')
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs, 
        background_seqs, 
        use_full_background_for_part_fn=(not partition_background_seqs)
    )
    for mo in progressively_fit_model(
            partitioned_and_coded_rnds_and_seqs, 
            ddg_array, ref_energy, 
            dna_conc, prot_conc
        ):
        
        if selex_db_conn != None:
            selex_db_conn.insert_model_into_db(
                mo.ref_energy, mo.ddg_array, mo.new_validation_lhd)
        
        if output_fname_prefix != None:
            ofname = "%s.FITMO.BSLEN%i.txt" % (
                output_fname_prefix, mo.ddg_array.motif_len)
            with open(ofname, "w") as ofp:
                write_output(
                    output_fname_prefix, mo.ddg_array, mo.ref_energy, ofp)
        break
    return
    
def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs
     ) = parse_arguments()
    fit_model(
        rnds_and_seqs, background_seqs, 
        initial_model,
        jolma_dna_conc, jolma_prot_conc,
        partition_background_seqs,
        selex_db_conn,
        initial_model.tf_name
    )
        
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
