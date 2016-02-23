import os

from collections import defaultdict, namedtuple
from itertools import chain
#from ENCODE_ChIPseq_tools import find_target_info
#    find_ENCODE_DCC_peaks, find_target_info )

################################################################################
#
# Insert ENCODE meta-data into the local database
#
################################################################################

def build_local_ENCODE_peak_fname(peak_id):
    from ENCODE_ChIPseq_tools import chipseq_peaks_base_dir
    cur = conn.cursor()
    query = """
    SELECT file_output_type, sample_type, bsid, encode_target_id 
      FROM encode_chipseq_peak_files, 
           encode_chipseq_experiments, 
           chipseq_targets 
     WHERE chipseq_targets.chipseq_target_id 
           = encode_chipseq_experiments.target 
       AND encode_chipseq_peak_files.encode_experiment_id 
           = encode_chipseq_experiments.encode_experiment_id
       AND encode_chipseq_peak_key = %s;
    """
    cur.execute(query, [peak_id,])
    res = cur.fetchall()
    if len(res) == 0: return None
    # encode_chipseq_peak_key is the primary key, so we shouldn't get multiple 
    # results
    assert len(res) == 1
    output_fname_template = "ChIPseq.{peak_type}.{sample_type}.{target_name}.ENCODECHIPSEQPK{peak_id}.bed.gz"
    peak_type, sample_type, bsid, encode_target_id = res[0]
    if peak_type == 'optimal idr thresholded peaks':
        peak_type = 'idr_optimal'
    elif peak_type == 'peaks' and bsid == 'merged':
        peak_type = 'replicate_merged'
    elif peak_type == 'peaks':
        peak_type = 'single_replicate'
    elif peak_type == 'hotspots':
        peak_type = 'hotspots'
    else:
        raise ValueError, "Unrecognized peak type: '%s'" % peak_type
    sample_type = sample_type.replace(" ", "__").replace("/", "-")
    target_name = encode_target_id.strip().split("/")[-2].replace("/", "-")
    return os.path.join( 
        chipseq_peaks_base_dir, 
        output_fname_template.format(
            peak_type=peak_type, sample_type=sample_type, 
            peak_id=peak_id, target_name=target_name) )

def sync_ENCODE_chipseq_peak_files():
    from ENCODE_ChIPseq_tools import download_and_index_peak_file
    # find all files in the database that don't have local copies
    cur = conn.cursor()
    query = """
    SELECT encode_chipseq_peak_key, remote_filename 
      FROM encode_chipseq_peak_files 
     WHERE local_filename is NULL;
    """
    cur.execute(query)
    all_peaks = cur.fetchall()
    for i, (peak_id, remote_filename) in enumerate(all_peaks):
        print "Processing peak_id '%i' (%i/%i)" % (peak_id, i, len(all_peaks))
        new_local_fname = build_local_ENCODE_peak_fname(peak_id)
        rv = download_and_index_peak_file(remote_filename, new_local_fname)
        if rv == 0:
            query = """
            UPDATE encode_chipseq_peak_files 
            SET local_filename = %s 
            WHERE encode_chipseq_peak_key = %s;
            """
            cur.execute(query, [new_local_fname, peak_id])
            conn.commit()
        else:
            raise ValueError, "Failed to sync encode TF peak id %s (%s)" % (
                peak_id, remote_filename)
    return

################################################################################
#
# Insert roadmap meta-data into the local database
#
################################################################################

def sync_roadmap_DNASE_peak_files():
    """Add local dnase files tot he DB. 
    
    This is just a stub to get some of the predictuion code working.
    """
    cur = conn.cursor()
    base_dir = "/mnt/data/epigenomeRoadmap/peaks/consolidated/narrowPeak/"
    for fname in os.listdir(base_dir):
        if "DNase.macs2" not in fname: continue
        sample_name = fname.split("-")[0]
        if int(sample_name[1:]) < 114: continue
        query = """
        INSERT INTO roadmap_dnase_files 
        (roadmap_sample_id, file_type, local_filename) 
        VALUES 
        (%s, 'optimal_peak', %s);
        """
        cur.execute(query, [sample_name, os.path.join(base_dir, fname)])
    conn.commit()
    return

################################################################################
#
# Get data from the local database
#
################################################################################
Genome = namedtuple('Genome', ['id', 'name', 'revision', 'species', 'filename'])
def load_genome_metadata(annotation_id):
    cur = conn.cursor()
    query = """
    SELECT name, revision, species, local_filename 
      FROM genomes 
     WHERE annotation_id=%s;
    """
    cur.execute(query, [annotation_id,])
    res = cur.fetchall()
    if len(res) == 0: 
        raise ValueError, \
            "No genome exists in the DB with annotation_id '%i' " \
                % annotation_id
    assert len(res) == 1
    return Genome(*([annotation_id,] + list(res[0])))

def load_genome_metadata_from_name(genome_name):
    cur = conn.cursor()
    query = """
    SELECT annotation_id 
      FROM genomes 
     WHERE name=%s;
    """
    cur.execute(query, [genome_name,])
    res = cur.fetchall()
    if len(res) == 0: 
        raise ValueError, \
            "No genome exists in the DB with name '%s' " \
                % genome_name
    assert len(res) == 1
    return load_genome_metadata(res[0][0])


def find_cisbp_tfids(species, tf_name, uniprot_ids, ensemble_ids):
    cur = conn.cursor()
    query = "SELECT tf_id FROM tfs WHERE dbid IN %s;"
    #print(cur.mogrify(
    #    query, [tuple(chain(uniprot_ids, chain(*ensemble_ids))),]))
    rv = []
    prot_and_gene_ids = tuple(chain(uniprot_ids, chain(*ensemble_ids)))
    if len(prot_and_gene_ids) > 0:
        cur.execute(query, [prot_and_gene_ids,]) 
        rv = [x[0] for x in cur.fetchall()]
    # if we can't find a reference from the uniprot or ensemble ids,
    # then try with the tf name
    if len(rv) != 1:
        query = "SELECT tf_id FROM tfs WHERE tf_species = %s and tf_name = %s;"
        res = cur.execute(
            query, (species, tf_name))
        rv = [x[0] for x in cur.fetchall()]
    
    # if we still an't find a match, try with the upper case
    if len(rv) != 1:
        query = "SELECT tf_id FROM tfs WHERE tf_species = %s and upper(tf_name) = upper(%s);"
        res = cur.execute(
            query, (species.replace(" ", "_"), tf_name))
        rv = [x[0] for x in cur.fetchall()]
        
    return rv

def encode_chipseq_exp_is_in_db(exp_id):
    """Check if a ENCODE ChIP-seq experiment has a datbase entry.

    returns True if the experiment is in the database, False if not
    """
    cur = conn.cursor()
    query = """
    SELECT encode_experiment_id 
      FROM encode_chipseq_experiments
     WHERE encode_experiment_id = %s;
    """
    cur.execute(query, [exp_id,])
    # if the experiment is already there, we are done
    res = cur.fetchall()
    assert len(res) in (0,1)
    if len(res) == 1:
        return True
    return False

def load_optimal_chipseq_peaks_and_matching_DNASE_files_from_db(
        tfid, annotation_id):
    cur = conn.cursor()    
    query = """
    SELECT roadmap_sample_id,
           dnase_peak_fname,
           chipseq_peak_fname
      FROM roadmap_matched_optimal_chipseq_peaks
     WHERE tf_id = %s
       AND annotation_id = %s;
    """
    rv = defaultdict(lambda: (set(), set()))
    cur.execute(query, [tfid, annotation_id])
    for sample_id, dnase_peak_fname, chipseq_peak_fname in cur.fetchall():
        rv[sample_id][0].add(chipseq_peak_fname)
        rv[sample_id][1].add(dnase_peak_fname)
    return rv

def encode_ChIPseq_peak_file_is_in_db(remote_filename):
    """Check if a ENCODE ChIPseq file has a database entry.

    returns True if the file is in the database, False if not
    """
    cur = conn.cursor()
    query = """
    SELECT * 
      FROM encode_chipseq_peak_files
     WHERE remote_filename ~ %s; 
    -- use a suffix match so that it will match even if 
    -- the http://encodeproject.org prefix is missing
    """
    cur.execute(query, [remote_filename,])
    # if the experiment is already there, we are done
    res = cur.fetchall()
    assert len(res) in (0,1)
    if len(res) == 1:
        return True
    return False

def load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        tfid, annotation_id):
    cur = conn.cursor()    
    query = """
    SELECT roadmap_sample_id,
           dnase_peak_fname,
           chipseq_peak_type,
           chipseq_peak_fname
      FROM roadmap_matched_chipseq_peaks
     WHERE tf_id = %s
       AND annotation_id = %s;
    """
    rv = defaultdict(lambda: (defaultdict(set), set()))
    cur.execute(query, [tfid, annotation_id])
    for ( sample_id, dnase_peak_fname, chipseq_peak_type, chipseq_peak_fname
           ) in cur.fetchall():
        rv[sample_id][0][chipseq_peak_type].add(chipseq_peak_fname)
        rv[sample_id][1].add(dnase_peak_fname)
    return rv

import psycopg2
conn = psycopg2.connect("host=mitra dbname=cisbp")
