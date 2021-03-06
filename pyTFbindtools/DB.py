import os

from collections import defaultdict, namedtuple
from itertools import chain
#from ENCODE_ChIPseq_tools import find_target_info
#    find_ENCODE_DCC_peaks, find_target_info )

################################################################################
#
# Find metadata about ENCODE ChIP-Seq Targets
#
################################################################################
import requests, json
import urllib2
import re
TargetInfo = namedtuple('TargetInfo', [
    'target_id', 
    'organism',
    'tf_name', 
    'uniprot_ids',
    'gene_name',
    'ensemble_ids',
    'cisbp_id'])

def find_target_info(target_id):
    URL = "https://www.encodeproject.org/{}?format=json".format(target_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    organism = response_json_dict['organism']['scientific_name'].replace(" ", "_")
    tf_name = response_json_dict['label']
    uniprot_ids = [x[10:] for x in response_json_dict['dbxref']
                   if x.startswith("UniProtKB:")]
    gene_name = response_json_dict['gene_name']
    ensemble_ids = sorted(
        get_ensemble_genes_associated_with_uniprot_id(uniprot_id) 
        for uniprot_id in uniprot_ids)
    cisbp_ids = find_cisbp_tfids(organism, tf_name, uniprot_ids, ensemble_ids)
    if len(cisbp_ids) == 0:
        cisbp_id = None
    else:
        assert len(cisbp_ids) == 1
        cisbp_id = cisbp_ids[0]
    rv = TargetInfo(target_id, organism, 
                    tf_name, uniprot_ids, 
                    gene_name, ensemble_ids, 
                    cisbp_id)
    return rv

def get_ensemble_genes_associated_with_uniprot_id(uniprot_id):
    ens_id_pat = '<property type="gene ID" value="(ENS.*?)"/>'
    res = urllib2.urlopen(
        "http://www.uniprot.org/uniprot/%s.xml" % uniprot_id)
    #print( res.read().decode('utf-8') )
    gene_ids = set(re.findall(ens_id_pat, res.read().decode('utf-8')))
    return sorted(gene_ids)

################################################################################
#
# Insert ENCODE meta-data into the local database
#
################################################################################
def find_or_insert_ENCODE_experiment(
        encode_exp_id, encode_sample_type, assay, encode_target_id=None):
    cur = conn.cursor()

    # first check to see if the experiment is already in the DB. If
    # so, we are done
    query = """
    SELECT encode_experiment_id 
      FROM encode_experiments 
     WHERE encode_experiment_id = %s;
    """
    cur.execute(query, [encode_exp_id,])
    # if the experiment is already there, we are done
    res = cur.fetchall()
    if len(res) == 1:
        return res[0][0]
        
    # find target id 
    if encode_target_id is not None:
        query = """
        SELECT target_id 
          FROM chipseq_targets
         WHERE encode_target_id = %s
        """
        cur.execute(query, [encode_target_id,])
        res = cur.fetchall()
    
        # if we can't find a matching tf id, insert it
        # XXX this section is untested
        if len(res) == 0:
            target_info = find_target_info(encode_target_id)
            query = "INSERT INTO chipseq_targets (encode_target_id, tf_id, organism, tf_name, uniprot_ids, ensemble_gene_ids) VALUES (%s, %s, %s, %s, %s, %s) RETURNING target_id"
            cur.execute(query, [encode_target_id, 
                                target_info.cisbp_id, 
                                target_info.organism,
                                target_info.tf_name,
                                target_info.uniprot_ids, 
                                target_info.ensemble_ids])
            res = cur.fetchall()
        assert len(res) == 1
        target_id = res[0][0]
    else:
        target_id = None
    
    # add the experiment data
    query = "INSERT INTO encode_experiments " \
          + "(encode_experiment_id, target_id, sample_type, assay) " \
          + "VALUES (%s, %s, %s, %s)"
    cur.execute(query, [
        encode_exp_id, target_id, encode_sample_type, assay])
    conn.commit()
    return encode_exp_id


def find_or_insert_experiment_from_called_peaks(called_peaks):
    cur = conn.cursor()

    # first check to see if the experiment is already in the DB. If
    # so, we are done
    encode_exp_ids = set(
        called_peak.exp_id for called_peak in called_peaks)
    assert len(encode_exp_ids) == 1, str(encode_exp_ids)
    encode_exp_id = encode_exp_ids.pop()
    query = """
    SELECT encode_experiment_id 
      FROM encode_chipseq_experiments 
     WHERE encode_experiment_id = %s;
    """
    cur.execute(query, [encode_exp_id,])
    # if the experiment is already there, we are done
    if len(cur.fetchall()) == 1:
        return
    # otherwise, insert everything
    encode_target_ids = set(
        called_peak.target_id for called_peak in called_peaks)
    assert len(encode_target_ids) == 1
    encode_target_id = encode_target_ids.pop()
    
    # find our associated target id 
    query = """
    SELECT chipseq_target_id 
      FROM chipseq_targets 
     WHERE encode_target_id = %s
    """
    cur.execute(query, [encode_target_id,])
    res = cur.fetchall()
    
    # if we can't find a matching tf id, insert it
    if len(res) == 0:
        target_info = find_target_info(encode_target_id)
        query = "INSERT INTO chipseq_targets (encode_target_id, tf_id, organism, tf_name, uniprot_ids, ensemble_gene_ids) VALUES (%s, %s, %s, %s, %s, %s) RETURNING chipseq_target_id"
        cur.execute(query, [encode_target_id, 
                            target_info.cisbp_id, 
                            target_info.organism,
                            target_info.tf_name,
                            target_info.uniprot_ids, 
                            target_info.ensemble_ids])
        res = cur.fetchall()
    assert len(res) == 1
    target_id = res[0][0]

    sample_types = set(
        called_peak.sample_type for called_peak in called_peaks)
    assert len(sample_types) == 1
    sample_type = sample_types.pop()
    # add the experiment data
    query = "INSERT INTO encode_chipseq_experiments " \
          + "(encode_experiment_id, target, sample_type) " \
          + "VALUES (%s, %s, %s)"
    cur.execute(query, [
        encode_exp_id, target_id, sample_type])
    return

def insert_chipseq_experiment_into_db(exp_id):
    """Download and insert ENCODE experiment metadata.
    """
    if encode_exp_is_in_db(exp_id):
        return
    
    called_peaks = list(find_called_peaks(exp_id, only_merged=False))
    if len(called_peaks) == 0: return
    
    # insert the experiment and target into the DB if necessary
    num_inserted = find_or_insert_experiment_from_called_peaks(called_peaks)
    cur = conn.cursor()
    for called_peak in called_peaks:
        # add the peak data
        query = """
        INSERT INTO encode_chipseq_peak_files
        (encode_experiment_id, bsid, rep_key, 
         file_format, file_format_type, file_output_type, remote_filename)
        VALUES 
        (%s, %s, %s, %s, %s, %s, %s)
        """
        try: 
            cur.execute(query, [
                called_peak.exp_id, called_peak.bsid, called_peak.rep_key,
                called_peak.file_format,
                called_peak.file_format_type,
                called_peak.output_type,
                called_peak.file_loc])
        except psycopg2.IntegrityError:
            print( "ERROR" )
            raise
            pass
    conn.commit()
    return


def insert_ENCODE_peak_file_into_db(file_data):
    """Insert ENCODE peak file into the database.
    """
    cur = conn.cursor()
    # add the peak data
    query = """
    INSERT INTO encode_chipseq_peak_files
    (encode_experiment_id, 
     bsid, 
     rep_key, 
     file_format, 
     file_format_type, 
     file_output_type, 
     remote_filename, 
     annotation)
    VALUES 
    (%s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING encode_chipseq_peak_key
    """
    assert file_data.assembly == 'hg19'
    try: 
        cur.execute(query, [
            file_data.exp_id, 
            file_data.bsid, 
            file_data.rep_key,
            file_data.file_format,
            file_data.file_format_type,
            file_data.output_type,
            file_data.file_loc,
            1])
    except psycopg2.IntegrityError:
        conn.rollback()
        print "ROLLBACK", file_data.exp_id
    else:
        res = cur.fetchall()
        assert len(res) == 1
        assert len(res[0]) == 1
        conn.commit()
        return res[0][0]
    return

def insert_ENCODE_bam_into_db(file_data):
    """Download and insert ENCODE experiment metadata.
    """
    cur = conn.cursor()
    # add the peak data
    query = """
    INSERT INTO encode_bam_files
    (annotation, encode_experiment_id, bsid, rep_key, remote_filename)
    VALUES 
    (%s, %s, %s, %s, %s)
    RETURNING bam_key
    """
    assert file_data.assembly == 'hg19'
    assembly_id = 1
    try: 
        cur.execute(query, [
            assembly_id, 
            file_data.exp_id, 
            file_data.bsid, 
            file_data.rep_key,
            "http://encodeproject.org" + file_data.file_loc
        ])
    except psycopg2.IntegrityError:
        conn.rollback()
        raise
        pass
    else:
        res = cur.fetchall()
        assert len(res) == 1
        assert len(res[0]) == 1
        conn.commit()
        return res[0][0]
    return

def build_local_ENCODE_bam_fname(bam_key):
    from ENCODE_ChIPseq_tools import chipseq_bams_base_dir
    cur = conn.cursor()
    query = """
    SELECT sample_type, bsid, encode_target_id, assay 
      FROM encode_bam_files NATURAL JOIN encode_experiments NATURAL LEFT JOIN chipseq_targets 
     WHERE bam_key = %s;
    """
    cur.execute(query, [bam_key,])
    res = cur.fetchall()
    if len(res) == 0: return None
    # encode_chipseq_peak_key is the primary key, so we shouldn't get multiple 
    # results
    assert len(res) == 1
    sample_type, bsid, encode_target_id, assay = res[0]
    sample_type = sample_type.replace(" ", "__").replace("/", "-")
    target_name = ( 
        'NULL' if encode_target_id is None 
        else encode_target_id.strip().split("/")[-2].replace("/", "-")
    )
    assay = assay.replace("-", "").upper()

    output_fname_template = "{assay}.{sample_type}.{target_name}.ENCODE{assay}BAM{bam_id}.bam"
    return os.path.join(
        chipseq_bams_base_dir,
        output_fname_template.format(
            assay=assay,
            sample_type=sample_type, 
            target_name=target_name, 
            bam_id=bam_key
        )
    )

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

    def download_and_insert_peak(peak_id, remote_filename):
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

    from grit.lib.multiprocessing_utils import run_in_parallel
    run_in_parallel(16, download_and_insert_peak, all_peaks) 
    #for i, (peak_id, remote_filename) in enumerate(all_peaks):
    #    print "Processing peak_id '%i' (%i/%i)" % (peak_id, i, len(all_peaks))
    #    download_and_insert_peak(peak_id, remote_filename)
    return

def sync_ENCODE_bam_files():
    from ENCODE_ChIPseq_tools import download_and_index_bam
    # find all files in the database that don't have local copies
    cur = conn.cursor()
    query = """
    SELECT bam_key, remote_filename 
      FROM encode_bam_files 
     WHERE local_filename is NULL;
    """
    cur.execute(query)
    all_peaks = cur.fetchall()
    def download_and_insert_peak(bam_id, remote_filename):
        global conn
        conn = psycopg2.connect("host=mitra dbname=cisbp")
        cur = conn.cursor()
        new_local_fname = build_local_ENCODE_bam_fname(bam_id)
        rv = download_and_index_bam(remote_filename, new_local_fname)
        print bam_id
        if rv == 0:
            query = """
            UPDATE encode_bam_files 
            SET local_filename = %s 
            WHERE bam_key = %s;
            """
            #print cur.mogrify(query, [new_local_fname, bam_id])
            cur.execute(query, [new_local_fname, bam_id])
            conn.commit()
        else:
            raise ValueError, "Failed to sync encode TF peak id %s (%s)" % (
                bam_id, remote_filename)

    from grit.lib.multiprocessing_utils import run_in_parallel
    run_in_parallel(4, download_and_insert_peak, all_peaks) 
    #for i, (bam_id, remote_filename) in enumerate(all_peaks):
    #    print "Processing peak_id '%i' (%i/%i)" % (bam_id, i, len(all_peaks))
    #    download_and_insert_peak(bam_id, remote_filename)
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

def load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, tfid=None, roadmap_sample_id=None, only_optimal=False):
    cur = conn.cursor()
    query = """
    SELECT roadmap_sample_id,
           tf_id,
           dnase_peak_fname,
           chipseq_peak_type,
           chipseq_peak_fname
      FROM {}
     WHERE annotation_id = %s""".format(
         'roadmap_matched_chipseq_peaks' if not only_optimal else (
             'roadmap_matched_optimal_chipseq_peaks')
     )
    args = [annotation_id,]
    if tfid is not None:
        query += """
       AND tf_id = %s"""
        args.append(tfid)
    if roadmap_sample_id is not None:
        query += """
       AND roadmap_sample_id = %s"""
        args.append(roadmap_sample_id)
    query += ';'
    rv = defaultdict(lambda: defaultdict(set))
    cur.execute(query, args)
    for ( sample_id, tf_id, dnase_peak_fname, chipseq_peak_type, chipseq_peak_fname
           ) in cur.fetchall():
        rv[sample_id][tf_id].add((chipseq_peak_type, chipseq_peak_fname))
        rv[sample_id]['dnase'].add(dnase_peak_fname)
    return rv

def load_optimal_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, tfid=None, roadmap_sample_id=None):
    return load_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, tfid, roadmap_sample_id, only_optimal=True)

def load_tf_names(tf_ids):
    cur = conn.cursor()
    query = "select tf_name from roadmap_matched_chipseq_peaks where tf_id = %s limit 1"
    tf_names = []
    for tf_id in tf_ids:
        #print tf_id
        cur.execute(query, [tf_id,])
        tf_names.append(cur.fetchall()[0][0])
    return tf_names

def load_tf_ids(tf_names):
    cur = conn.cursor()
    query = "select tf_id from roadmap_matched_chipseq_peaks where tf_name = %s limit 1"
    tf_ids = []
    for tf_name in tf_names:
        cur.execute(query, [tf_name,])
        res = cur.fetchall()
        if len(res) == 0: 
            raise ValueError, "Can not find TFID for factor name '%s'" % tf_name
        assert len(res) == 1
        tf_ids.append(res[0][0])
    return tf_ids

def load_tf_names_for_sample(sample_id):
    cur = conn.cursor()
    query = "select distinct tf_name from roadmap_matched_chipseq_peaks where roadmap_sample_id = %s"
    cur.execute(query, [sample_id,])
    print cur.mogrify(query, [sample_id,])
    tf_ids = [x[0] for x in cur.fetchall()]
    return tf_ids

def load_ENCODE_target_id(tf_ids):
    cur = conn.cursor()
    query = "select ENCODE_target_id from chipseq_targets where tf_id = %s"
    encode_target_ids = []
    for tf_id in tf_ids:
        cur.execute(query, [tf_id,])
        res = cur.fetchall()
        if len(res) == 0:
            raise ValueError, "TFID '%s' does not appear in DB" % tf_id
        encode_target_ids.append(res[0][0])
    return encode_target_ids

def load_dnase_fnames(roadmap_sample_id):
    cur = conn.cursor()
    query = """
        SELECT dnase_bam_fname
          FROM roadmap_matched_dnase_bams
         WHERE roadmap_sample_id=%s;
    """
    fnames = []
    cur.execute(query, [roadmap_sample_id,])
    fnames.append(cur.fetchall()[0][0])
    return fnames

def load_chipseq_fnames(roadmap_sample_id, tf_id):
    conn = psycopg2.connect("host=mitra dbname=cisbp")
    cur = conn.cursor()
    query = """
        SELECT chipseq_bam_fname, control_bam_fname
          FROM roadmap_matched_chipseq_bams
         WHERE roadmap_sample_id=%s
           AND tf_id=%s
    """
    fnames = []
    control_fnames = []
    cur.execute(query, [roadmap_sample_id, tf_id])
    #print cur.mogrify(query, [roadmap_sample_id, tf_id])
    fnames, control_fnames = zip(*cur.fetchall())
    return fnames, control_fnames

import psycopg2
conn = psycopg2.connect("host=mitra dbname=cisbp")
