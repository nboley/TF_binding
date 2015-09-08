import os, sys

import psycopg2

#from fit_selex import estimate_dg_matrix_with_adadelta, find_pwm, load_sequences

def insert_model_into_db(exp_id, motif_len, 
                         ref_energy, ddg_array, 
                         chem_affinities, 
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
    consensus_energy = ddg_array.calc_min_energy(ref_energy)
    base_contributions = ddg_array.calc_base_contributions()
    
    print exp_id, bs_len, ref_energy, ddg_array,chem_affinities, lhd_hat, lhd_path
    assert False
    return

def fit_model(exp_id, rnds_and_seqs, ddg_array, ref_energy):
    for rnd_num in xrange(
            min(20-motif_len+1, 
                len(rnds_and_seqs[0][0])-ddg_array.motif_len+1)):
        bs_len = ddg_array.motif_len
        partitioned_and_coded_rnds_and_seqs = PartitionedAndCodedSeqs(
            rnds_and_seqs, bs_len)

        pyTFbindtools.log("Estimating energy model", 'VERBOSE')
        ( ddg_array, ref_energy, lhd_path, lhd_hat 
            ) = estimate_dg_matrix_with_adadelta(
                partitioned_and_coded_rnds_and_seqs,
                ddg_array, ref_energy,
                dna_conc, prot_conc)
        insert_model_into_db(exp_id, bs_len, ref_energy, ddg_array,
                             chem_affinities, lhd_hat, lhd_path)
        
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
            
    return

def get_fnames(exp_id):
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()
    query = """
    SELECT fname 
      FROM selex_round
     WHERE selex_exp_id = '%i';
    """
    res = cur.execute(query % exp_id).fetchall()
    print res
    assert False

def main():
    exp_id = int(sys.argv[1])
    selex_fnames = get_fnames(exp_id)
    rnds_and_seqs = load_sequences(x.name for x in selex_fnames)
    return
    factor_name = 'TEST'
    initial_binding_site_len = 6
    pwm = find_pwm(rnds_and_seqs, initial_binding_site_len)
    motif = Motif("aligned_%imer" % args.initial_binding_site_len, 
                      factor_name, pwm)
    ref_energy, ddg_array = motif.build_ddg_array()
    ddg_array_hat, ref_energy_hat = fit_model(
        rnds_and_seqs, ddg_array, ref_energy )
    return

main()
