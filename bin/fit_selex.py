import os, sys
import gzip
import random

from itertools import izip, chain
from collections import defaultdict, namedtuple

import numpy as np

from pyDNAbinding.binding_model import (
    EnergeticDNABindingModel, PWMBindingModel, load_binding_model, 
    DNABindingModels )
from pyDNAbinding.misc import optional_gzip_open, load_fastq

import pyTFbindtools

import pyTFbindtools.selex

from pyTFbindtools.selex import (
    PartitionedAndCodedSeqs, 
    progressively_fit_model, find_pwm, sample_random_seqs,
    estimate_chem_affinities_for_selex_experiment)

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
n_prot_mol = n_dna_seq/25
n_water_mol = 55.1*5.0e-5*6.02e23 # mol/L*volume*molecule/mol

#jolma_dna_conc = n_dna_seq/(6.02e23*5.0e-5) # mol/L
#jolma_prot_conc = jolma_dna_conc/25 # mol/L (should be 25)

jolma_dna_conc = n_dna_seq/(n_dna_seq + n_prot_mol + n_water_mol)
jolma_prot_conc = n_prot_mol/(n_dna_seq + n_prot_mol + n_water_mol)

#prot_conc *= 10

USE_FULL_BG_FOR_PART_FN = True
DEFAULT_MIN_NUM_BG_SEQS = 100000

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

def initialize_starting_motif(
        pwm_fp, 
        energy_mo_fp, 
        rnds_and_seqs,
        initial_binding_site_len,
        include_shape):
    assert (pwm_fp is None) or (energy_mo_fp is None), \
        "Cant initialize a motif from both a pwm and energy model"
    if pwm_fp is not None:
        return load_binding_model(pwm_fp.name).build_energetic_model(
            include_shape=include_shape)
    elif energy_mo_fp is not None:
        return load_binding_model(energy_mo_fp.name)
    else:
        pyTFbindtools.log(
            "Initializing starting location from %imer search" % initial_binding_site_len, 
            'VERBOSE')
        bs_len = initial_binding_site_len
        pwm = find_pwm(rnds_and_seqs, initial_binding_site_len)
        pwm_model = PWMBindingModel(pwm)
        rv = pwm_model.build_energetic_model(include_shape=include_shape)
        return rv
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
    pwm = build_pwm(ddg_array, ref_energy, -19.0)
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
    
    parser.add_argument( '--initial-binding-site-len', type=int, default=7,
        help='The starting length of the binding site (this will grow)')
    parser.add_argument( '--include-shape', default=False, action='store_true',
        help='Use shape features when estimating hte binding affinity')

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
        args.include_shape)
    # close the starting motif files
    if args.starting_pwm is not None: 
        args.starting_pwm.close()
    if args.starting_energy_model is not None: 
        args.starting_energy_model.close()
        
    return ( motif, 
             rnds_and_seqs, background_seqs, 
             selex_db_conn, 
             args.partition_background_seqs,
             args.ofname_prefix)

def fit_model(rnds_and_seqs, background_seqs, 
              initial_model, 
              dna_conc, prot_conc,
              partition_background_seqs,
              selex_db_conn=None,
              output_fname_prefix=None):
    assert selex_db_conn is not None or output_fname_prefix is not None

    chem_affinities = estimate_chem_affinities_for_selex_experiment(
        background_seqs, 
        max(rnds_and_seqs.keys()), 
        initial_model, dna_conc, prot_conc)

    pyTFbindtools.log("Coding sequences", 'VERBOSE')
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs, 
        background_seqs, 
        include_shape_features = (
            True if initial_model.encoding_type == 'ONE_HOT_PLUS_SHAPE' 
            else False),
        use_full_background_for_part_fn=(not partition_background_seqs)
    )
    model_meta_data = dict(initial_model.meta_data.iteritems())
    fit_models = []
    for mo in progressively_fit_model(
            partitioned_and_coded_rnds_and_seqs, 
            initial_model, chem_affinities,
            dna_conc, prot_conc
        ):
        
        if selex_db_conn != None:
            selex_db_conn.insert_model_into_db(
                mo.energetic_model, mo.new_validation_lhd)
        model_meta_data['dna_conc'] = dna_conc
        model_meta_data['prot_conc'] = prot_conc
        
        #model_meta_data['lhd_path'] = mo.lhd_path.tolist()
        model_meta_data['lhd_hat'] = float(mo.lhd_hat)
        model_meta_data['prev_validation_lhd'] = float(mo.prev_validation_lhd)
        model_meta_data['new_validation_lhd'] = float(mo.new_validation_lhd)

        fit_models.append(mo.energetic_model)
        if output_fname_prefix != None:
            ofname = "%s.FITMO.BSLEN%i.yaml" % (
                output_fname_prefix, mo.energetic_model.ddg_array.motif_len)
            with open(ofname, "w") as ofp:
                mo.energetic_model.save(ofp)

    if output_fname_prefix is not None:
        fit_models = DNABindingModels(fit_models)
        with open("%s.yaml" % output_fname_prefix, "w") as fp:
            fit_models.save(fp)
    return
    
def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs,
      ofname_prefix
     ) = parse_arguments()


    """    
    partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
        rnds_and_seqs, 
        background_seqs, 
        use_full_background_for_part_fn=True,
        n_partitions = 4
    )

    print x0
    data = partitioned_and_coded_rnds_and_seqs.validation
    from pyTFbindtools.selex.log_lhd import calc_log_lhd
    calc_lhd, calc_grad = calc_lhd_factory(data)
    chem_affinities = np.array(
        [-28.1]*2, dtype='float32')
    print calc_lhd(x0.T, chem_affinities)
    return
    """
    fit_model(
        rnds_and_seqs, background_seqs, 
        initial_model,
        jolma_dna_conc, jolma_prot_conc,
        partition_background_seqs,
        selex_db_conn,
        ofname_prefix
    )
        
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
    return

if __name__ == '__main__':
    main()
