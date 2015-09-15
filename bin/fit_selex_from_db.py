import os, sys

import psycopg2

import numpy as np

from fit_selex import (
    estimate_dg_matrix_with_adadelta, find_pwm, load_sequences, 
    Motif, PartitionedAndCodedSeqs, pyTFbindtools, find_best_shift, DeltaDeltaGArray )
import fit_selex
fit_selex.pyTFbindtools.VERBOSE = True
fit_selex.pyTFbindtools.DEBUG = True


def insert_model_into_db(exp_id, motif_len, 
                         ref_energy, ddg_array, 
                         validation_lhd, lhd_path):
    """
                Table "public.selex_models"
          Column      |        Type        | Modifiers 
    ------------------+--------------------+-----------
     selex_exp_id     | integer            | not null
     motif_len        | integer            | not null
     consensus_energy | double precision   | not null
     ddg_array        | double precision[] | not null
     chem_affinities  | double precision[] | 
     validation_lhd   | double precision   | 
     test_lhd_path    | double precision[] | 
    Foreign-key constraints:
        "selex_models_selex_exp_id_fkey" FOREIGN KEY (selex_exp_id) 
            REFERENCES selex_experiments(selex_exp_id)
    """
    motif_len = ddg_array.motif_len
    consensus_energy, base_contributions = ddg_array.calc_normalized_base_conts(ref_energy)
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
    INSERT INTO selex_models (selex_exp_id, motif_len, consensus_energy, ddg_array, validation_lhd, test_lhd_path) VALUES (%s, %s, %s, %s, %s, %s) RETURNING key
    """
    cur.execute(query, (
        exp_id, motif_len, float(consensus_energy), base_contributions.tolist(),
        float(validation_lhd), lhd_path))
    conn.commit()
    conn.close()
    return

def fit_model(exp_id, rnds_and_seqs, ddg_array, ref_energy, dna_conc, prot_conc):
    while True:
        bs_len = ddg_array.motif_len
        partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
            rnds_and_seqs, bs_len)

        ( ddg_array, ref_energy, lhd_path, lhd_hat 
            ) = estimate_dg_matrix_with_adadelta(
                partitioned_and_coded_rnds_and_seqs,
                ddg_array, ref_energy,
                dna_conc, prot_conc)
        insert_model_into_db(exp_id, bs_len, ref_energy, ddg_array,
                             lhd_hat, lhd_path)
        
        if ( bs_len >= 20 
             or bs_len+1 >= partitioned_and_coded_rnds_and_seqs.seq_length):
            break

        shift_type = find_best_shift(rnds_and_seqs, ddg_array, ref_energy)
        if shift_type == 'LEFT':
            ddg_array = np.insert(ddg_array, 0, np.zeros(3, dtype='float32')
                              ).view(DeltaDeltaGArray)
        elif shift_type == 'RIGHT':
            ddg_array = np.append(ddg_array, np.zeros(3, dtype='float32')).view(
                DeltaDeltaGArray)
        else:
            assert False, "Unrecognized shift type '%s'" % shift_type
        print "Finished model w/ length %i" % bs_len
    
    return

def get_fnames(exp_id):
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()
    query = """
    SELECT rnd, fname 
      FROM selex_round
     WHERE selex_exp_id = '%i'
     ORDER BY rnd;
    """
    cur.execute(query % exp_id)
    res = [x[1] for x in cur.fetchall()]
    return res

def get_dna_and_prot_conc(exp_id):
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()
    query = """
    SELECT dna_conc, prot_conc 
      FROM selex_round
     WHERE selex_exp_id = '%i';
    """
    cur.execute(query % exp_id)
    concs = set(cur.fetchall())
    assert len(concs) == 1
    return concs.pop()

def get_experiment_to_process():
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
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
        return exp_id

def main():
    pyTFbindtools.VERBOSE = True
    #pyTFbindtools.DEBUG_VERBOSE = True
    #exp_id = int(sys.argv[1])
    while True:
        exp_id = get_experiment_to_process()
        if exp_id == None: break
        print "Processing experiment %i" % exp_id
        selex_fnames = get_fnames(exp_id)
        dna_conc, prot_conc = get_dna_and_prot_conc(exp_id)
        rnds_and_seqs = load_sequences(selex_fnames)
        initial_binding_site_len = 6
        pwm = find_pwm(rnds_and_seqs, initial_binding_site_len)
        motif = Motif('SELEXexp%i' % exp_id, str(exp_id), pwm)
        ref_energy, ddg_array = motif.build_ddg_array()
        fit_model(
            exp_id, rnds_and_seqs, ddg_array, ref_energy, dna_conc, prot_conc )
    return

main()
