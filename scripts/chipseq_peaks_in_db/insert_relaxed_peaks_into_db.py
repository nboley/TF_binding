import os, sys
import re

from collections import namedtuple, defaultdict

import requests, json

from pyTFbindtools import DB
from pyTFbindtools.ENCODE_ChIPseq_tools import find_ENCODE_DCC_experiment_metadata
cur = DB.conn.cursor()

MERGED_PEAKS_BASE_DIR = "/mnt/data/ENCODE/peaks_spp/mar2012/distinct/combrep/regionPeak/"
OUTPUT_MERGED_PEAKS_BASE_DIR = "/mnt/data/TF_binding/in_vivo/ENCODE/anshuls_TF_peaks/combrep/pos_sorted/"

def load_merged_peak_fnames(path=MERGED_PEAKS_BASE_DIR):
    fnames = []
    for fname in os.listdir(path):
        # skip histone marks
        if fname.startswith('wgEncodeBroadHistone'):
            continue
        fnames.append(fname)
    return fnames

def load_metadata(fname='ENCODE.hg19.TFBS.metadata.jun2012.SPP_pooled.tsv'):
    MetaData = None
    all_data = {}
    with open(fname) as fp:
        for line_num, line in enumerate(fp):
            data = line.strip().split("\t")
            # process the header
            if line_num == 0: 
                header = [re.sub("\W+", "_", x) for x in data][:-2]
                MetaData = namedtuple('MetaData', header)
            else:
                data = MetaData(*data)
                assert data.FILENAME not in all_data
                all_data[data.FILENAME] = data
    return all_data

cached_sample_types = {
    'ECC-1': 'Ishikawa',
    'HRE': 'kidney epithelial cell',
    'HMF': 'fibroblast of mammary gland',
    'Fibrobl': 'skin fibroblast',
    'HFF': 'foreskin fibroblast',
    'Gliobla': 'H54'
}
with open("celltype_sample_mapping.txt") as fp:
    for line in fp:
        celltype, sample_type = line.strip().split("\t")
        cached_sample_types[celltype] = sample_type

def find_sample_type(metadata):
    if metadata.CELLTYPE in cached_sample_types:
        return cached_sample_types[metadata.CELLTYPE]
    URL = "https://www.encodeproject.org/search/?searchTerm={}&type=biosample&organism.scientific_name=Homo%20sapiens&format=json&limit=all&frame=object".format(metadata.CELLTYPE)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    biosample_data = set()
    for biosample in response_json_dict['@graph']:
        biosample_data.add((
            biosample['biosample_term_name'],
            tuple(biosample['dbxrefs'])
        ))
    biosample_term_names = sorted(set(str(x[0]) for x in biosample_data))
    biosample_term_name = None
    assert len(biosample_term_names) > 0
    if len(biosample_term_names) == 1:
        biosample_term_name = biosample_term_names[0]
    elif len(biosample_term_names) > 1:
        # check if there is an exact match. If there is, return it
        if biosample_term_name == None and metadata.CELLTYPE in biosample_term_names:
            biosample_term_name = metadata.CELLTYPE
        if biosample_term_name == None:
            for name, aliases in biosample_data:
                if 'UCSC-ENCODE-cv:%s' % metadata.CELLTYPE in aliases:
                    biosample_term_name = name
                    print "here", metadata.CELLTYPE, biosample_data
                    break
        if biosample_term_name == None:
            assert False

    cached_sample_types[metadata.CELLTYPE] = biosample_term_name

    #assert False
    

def find_chipseq_target_id(metadata):
    # skip low quality data sets
    if metadata.INTEGRATED_QUALITY_FLAG == '-1':
        assert False
    
    # Special case some of the differences
    if metadata.HGNC_TARGET_NAME == 'POLR2A' and metadata.TARGET_NAME == 'Pol2':
        return 91
    if metadata.HGNC_TARGET_NAME == 'JUND' and metadata.TARGET_NAME == 'JunD':
        return 108
    if metadata.HGNC_TARGET_NAME == 'JUND' and metadata.TARGET_NAME == 'eGFP-JunD':
        return 76
    if metadata.HGNC_TARGET_NAME == 'GATA2' and metadata.TARGET_NAME == 'GATA-2':
        return 127
    if metadata.HGNC_TARGET_NAME == 'GATA2' and metadata.TARGET_NAME == 'GATA2_(SC-267)':
        return 127
    if metadata.HGNC_TARGET_NAME == 'FOS' and metadata.TARGET_NAME == 'c-Fos':
        return 141
    if metadata.HGNC_TARGET_NAME == 'GABPB1' and metadata.TARGET_NAME == 'GABP':
        return 205    
    if metadata.HGNC_TARGET_NAME == 'NFKB1' and metadata.TARGET_NAME == 'NFKB':
        return 134

    # try to construct the target id from the tf name
    encode_target_id = "/targets/%s-human/" % metadata.TARGET_NAME
    query = "SELECT chipseq_target_id FROM chipseq_targets WHERE encode_target_id = %s"
    cur.execute(query, [encode_target_id,])
    res = cur.fetchall()
    if len(res) > 0:
        # the target id is unique
        assert len(res) == 1
        return res[0][0]

    # try to match an ensemble id
    query = "SELECT chipseq_target_id FROM chipseq_targets WHERE %s = ANY(ensemble_gene_ids)"
    cur.execute(query, [metadata.ENSEMBL_TARGET_ID,])
    res = cur.fetchall()
    if len(res) > 0:
        # if we have multiple results, do this by hand
        if len(res) > 1: return None
        assert len(res) == 1
        return res[0][0]
        
    print metadata
    assert False
    return None

lab_id_mapping = {
    'HudsonAlpha': '/labs/richard-myers/',
    'Stanford': '/labs/michael-snyder/',
    'UW': '/labs/john-stamatoyannopoulos/',
    'USC': '/labs/peggy-farnham/',
    'Harvard': '/labs/kevin-struhl/',
    'UT-A': '/labs/vishwanath-iyer/',
    'Yale': '/labs/michael-snyder/',
    'UChicago': '/labs/kevin-white/'
}

def load_exp_id_mapping(fname="ENCODE_ID_MAPPING.txt"):
    rv = {}
    with open(fname) as fp:
        for line in fp:
            prefix, exp_id = line.split()
            assert prefix not in exp_id
            rv[prefix] = exp_id
    return rv

def find_experiment_id(metadata):
    target_id = find_chipseq_target_id(metadata)
    assert target_id != None
    sample_type = find_sample_type(metadata)
    assert sample_type != None
    query = """
    SELECT encode_experiment_id 
      FROM encode_chipseq_experiments 
     WHERE target=%s 
       AND sample_type=%s
    """
    cur.execute(query, [target_id, sample_type])
    valid_experiment_ids = []
    for (exp_id,) in cur.fetchall():
        ( lab_metadata, description, treatment
            ) = find_ENCODE_DCC_experiment_metadata(exp_id)
        
        lab_id = lab_metadata['@id']
        if lab_id != lab_id_mapping[metadata.LAB]: continue
        if metadata.TREATMENT == 'None' and treatment != None: continue
        if metadata.TREATMENT != 'None' and treatment == None: continue
        if metadata.PROTOCOL != 'std' and description.find(metadata.PROTOCOL) == -1:
            continue
        valid_experiment_ids.append((exp_id, description))
    if len(valid_experiment_ids) == 1:
        return valid_experiment_ids[0][0]
    if len(valid_experiment_ids) > 1:
        #print metadata.PROTOCOL, metadata.TREATMENT, description
        return valid_experiment_ids
    if len(valid_experiment_ids) == 0:
        print metadata.TARGET_NAME, sample_type
        return None
    
def infer_experiment_id(metadata):
    # dead code for inferring the experiment ID
    # I've already run this, and then cleaned up 
    # the results by hand
    experiment_data = find_experiment_id(metadata)
    if isinstance(experiment_data, str):
        print "+", fname_key, experiment_data
    elif experiment_data is None:
        print metadata
        print "-", fname_key
    elif isinstance(experiment_data, list):
        print "0", fname
        print "0", metadata.LAB, metadata.PROTOCOL, metadata.TREATMENT
        for exp_id, desc in experiment_data:
            print "0", exp_id, desc
        assert False

def insert_anshul_merged_peak_into_db(exp_id, local_fname):
    """
     encode_experiment_id    | text    | not null
     bsid                    | text    | not null # merged
     rep_key                 | text    | not null # merged
     file_format             | text    | not null # bed
     file_format_type        | text    | not null # narrowPeak
     file_output_type        | text    | not null # anshul relaxed ranked peaks 
     remote_filename         | text    | 
     local_filename          | text    | 
     encode_chipseq_peak_key | SERIAL  | not null #
     annotation              | integer | not null # 
    """
    query = """
    INSERT INTO encode_chipseq_peak_files
    ( encode_experiment_id, bsid, rep_key, 
      file_format, file_format_type, file_output_type, 
      local_filename,
      annotation ) VALUES (
      %s, 'merged', 'merged',
      'bed', 'narrowPeak', 'anshul relaxed ranked peaks',
      %s,
      1
    );
    """
    assert os.path.isfile(local_fname)
    cur.execute(query, [exp_id, local_fname])
    return

def insert_experiment_into_db(exp_id, target_id, sample_type):
    """
     encode_experiment_id | text    | not null
     target               | integer | not null
     sample_type          | text    | not null
    """

    # check if it already exists
    query = """
    SELECT target, sample_type 
      FROM encode_chipseq_experiments 
    WHERE encode_experiment_id = %s
    """
    cur.execute(query, [exp_id, ])
    res = cur.fetchall()
    # if it does exist, make sure that everything matches
    if len(res) > 0:
        assert len(res) == 1
        res = res[0]
        print res, target_id, sample_type
        assert res[0] == target_id
        assert res[1] == sample_type
    # if it doesn't exist, insert it
    else:
        query = "INSERT INTO encode_chipseq_experiments VALUES (%s, %s, %s)"
        cur.execute(query, [exp_id, target_id, sample_type])
    return

def main():
    all_metadata = load_metadata()
    merged_peak_fnames = load_merged_peak_fnames()
    exp_id_mapping = load_exp_id_mapping()
    fnames_missing_metadata = []
    for fname in merged_peak_fnames:
        fname_key =  fname.split(".bam")[0]
        if fname_key not in all_metadata:
            fnames_missing_metadata.append(fname)
        else:
            metadata = all_metadata[fname_key]
            # skip low quality experiments
            if metadata.INTEGRATED_QUALITY_FLAG == '-1':
                continue

            target_id = find_chipseq_target_id(metadata)
            assert target_id != None
            sample_type = find_sample_type(metadata)
            assert sample_type != None
            exp_id = exp_id_mapping[fname_key]
            assert exp_id != None
            print "\t".join((fname_key, exp_id, str(target_id), sample_type))
            #insert_experiment_into_db(exp_id, target_id, sample_type)
            new_fname = (
                OUTPUT_MERGED_PEAKS_BASE_DIR 
                + fname.replace(".regionPeak.gz", ".narrowPeak.bgz") )
            insert_anshul_merged_peak_into_db(exp_id, new_fname)
    DB.conn.commit()
    ## Cache the cell types
    #with open("celltype_sample_mapping.txt", "w") as ofp: 
    #    for celltype, sample_type in cached_sample_types.items():
    #        ofp.write("%s\t%s\n" % (celltype, sample_type))
    
            
    
if __name__ == '__main__':
    main()
