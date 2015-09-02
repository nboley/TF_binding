import os, sys
from collections import defaultdict
import psycopg2

aliases = {
    'ZNF306': 'ZKSCAN3',
    'ZNF435': 'ZSCAN16',
    'Tcfap2a': 'TFAP2A',
    'HOXA10': None, # Dont know why this is missing
    'Bhlhb2': 'BHLHE40',
    'Rhox11': None, # Mouse TF
    'Msx3': None, # ??
    'CART1': 'ALX1',
    'Bhlhb2': 'BHLHE40',
    'BHLHB3': 'BHLHE41',
    'POU5F1P1': 'POU5F1B',
    'HINFP1': 'HINFP',
    'BHLHB2': 'BHLHE40',
    'RAXL1': 'RAX2',
    'Zfp740': 'ZNF740',
    'Zfp652': 'ZNF652',
}

def parse_input_files(base_dir):
    selex_files = defaultdict(list)
    for fname in os.listdir(base_dir):
        if not fname.endswith("fastq.gz"): continue
        # skip the control sequencing
        if fname.startswith('ZeroCycle'):
            continue
        # Skip the ChIP-seq fastqs
        if fname.startswith('LoVo'):
            continue
        data = fname.split("_")
        factor = data[0]
        primer = data[1]
        rnd = int(data[-1].split(".")[0])
        selex_files[(factor, primer)].append(
            (rnd, os.path.join(base_dir, fname)))
    return selex_files

def main():
    selex_dir = "/mnt/data/TF_binding/in_vitro/HT_SELEX/"
    selex_files = parse_input_files(selex_dir)

    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()
    
    for (factor, primer), rnds_and_fnames in selex_files.items():
        alias = aliases[factor] if factor in aliases else factor
        if alias == None: continue
        query = "select tf_id from tfs where tf_species = 'Homo_sapiens' and tf_name = '%s'" % alias.upper()
        cur.execute(query)
        db_results = cur.fetchall()
        assert len(db_results) == 1
        tf_id = db_results[0][0]
        query = "INSERT INTO selex_experiments (tf_id) VALUES ('%s') RETURNING selex_exp_id" % tf_id
        cur.execute(query)
        selex_exp_id = cur.fetchall()[0][0]
        for rnd, fname in sorted(rnds_and_fnames):
            query = "INSERT INTO selex_round VALUES (%i, %i, '%s', 7.5e-8, 3e-09, '%s')" % (selex_exp_id, rnd, primer, fname)
            cur.execute(query)
    conn.commit()
    return

main()    
