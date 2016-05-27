import os, sys, time

import requests, json

from warnings import warn

import urllib2

from itertools import chain

from collections import defaultdict, namedtuple

import re

import gzip, io

from multiprocessing import Value

import psycopg2

from DB import conn, find_cisbp_tfids, insert_ENCODE_bam_into_db, \
    find_or_insert_ENCODE_experiment, insert_ENCODE_peak_file_into_db
ExperimentFile = namedtuple('ExperimentFile', [
    'exp_id', 'assay',
    'target_id', 
    'sample_type', 'rep_key', 'bsid', 
    'paired_end', 'paired_with',
    'assembly',
    'file_format', 'file_format_type', 'output_type', 'file_loc'])

BASE_URL = "http://encodeproject.org/"

chipseq_peaks_base_dir = "/mnt/lab_data/kundaje/users/nboley/TF_binding/ENCODE_ChIPseq_peaks/"
chipseq_bams_base_dir = "/mnt/lab_data/kundaje/users/nboley/TF_binding/ENCODE_ChIPseq_bams/"

################################################################################
#
# Find metadata associated with a particular ENCODE experiment ID
#
################################################################################
def find_ENCODE_DCC_experiment_metadata(experiment_id):
    """Find metadata associated with an ENCODE experiment.

    Arguments:
    experiment_id: ENCODE experiment to find files for. 
    Returns: labmetadata, description, num_treatments
    """
    URL = "https://www.encodeproject.org/experiments/{}/".format(experiment_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    treatments = sorted(
        x['library']['biosample']['treatments']
        for x in response_json_dict['replicates'] )
    treatment_term_names = set()
    for treatment in treatments:
        treatment_term_names.update(x['treatment_term_name'] for x in treatment)
    return ( response_json_dict['lab'], 
             response_json_dict['description'], 
             (sorted(treatment_term_names) 
              if len(treatment_term_names) > 0 
              else None)
            )

def find_ENCODE_DCC_experiment_controls(experiment_id):
    """Find metadata associated with an ENCODE experiment.

    Arguments:
    experiment_id: ENCODE experiment to find files for. 
    Returns: labmetadata, description, num_treatments
    """
    URL = "https://www.encodeproject.org/experiments/{}/".format(experiment_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    return [x['accession'] for x in response_json_dict['possible_controls']]

################################################################################
#
# Find files associated with a particular ENCODE experiment ID
#
################################################################################


def find_ENCODE_DCC_experiment_files(experiment_id):
    """Iterate over files associated with an ENCODE experiment.

    Arguments:
    experiment_id: ENCODE experiment to find files for. 
    Returns: iterator yielding ExperimentFile objects
    """
    URL = "https://www.encodeproject.org/experiments/{}/".format(experiment_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()

    assay = response_json_dict['assay_term_name']

    try: target_id = response_json_dict['target']['@id']
    except KeyError: target_id = None

    sample_type = response_json_dict['biosample_term_name']

    # build the replicate mapping
    replicates = {}
    for rep_rec in response_json_dict['replicates']:
        key = (rep_rec['biological_replicate_number'], 
               rep_rec['technical_replicate_number'])
        replicates[key] = rep_rec

    for file_rec in response_json_dict['files']:
        file_format = file_rec['file_format']
        try: file_format_type = file_rec['file_format_type']
        except KeyError: file_format_type = None
        output_type = file_rec['output_type']

        if 'replicate' not in file_rec: 
            rep_key = 'merged'
            bsid = 'merged'
        else:
            rep_key = (file_rec['replicate']['biological_replicate_number'],
                       file_rec['replicate']['technical_replicate_number'] )
            bsid = replicates[rep_key]['library']['biosample']['accession']
        file_loc = file_rec['href']
        try: assembly = file_rec['assembly']
        except KeyError: assembly = None 
        try: read_pair = file_rec['paired_end']
        except KeyError: read_pair=None
        try: paired_with = file_rec['paired_with']
        except KeyError: paired_with=None
        yield ExperimentFile(
            experiment_id, assay,
            target_id, 
            sample_type, rep_key, bsid,
            read_pair, paired_with,
            assembly,
            file_format, file_format_type, output_type, file_loc)

    return 

def find_ENCODE_DCC_bams(experiment_id):
    """Iterate over all bams associated with an ENCODE experiment.

    Arguments:
    experiment_id: ENCODE experiment to find bams for. 
    Returns: iterator yielding ExperimentFile objects
    """
    for experiment_file in find_ENCODE_DCC_experiment_files(experiment_id):
        if experiment_file.file_format == 'bam':
            yield experiment_file
    return

def find_ENCODE_DCC_fastqs(experiment_id):
    """Iterate over all bams associated with an ENCODE experiment.

    Arguments:
    experiment_id: ENCODE experiment to find bams for. 
    Returns: iterator yielding ExperimentFile objects
    """
    single_end = {}
    pair1 = {}
    pair2 = {}
    try: 
        experiment_files = list(find_ENCODE_DCC_experiment_files(experiment_id))
    except Exception:
        print >> sys.stderr, experiment_id
        raise
    for experiment_file in experiment_files:
        if experiment_file.file_format == 'fastq':
            if experiment_file.paired_end is None:
                single_end[experiment_file.file_loc.split("/")[2]] = experiment_file
            elif experiment_file.paired_end == '1':
                pair1[experiment_file.file_loc.split("/")[2]] = experiment_file
            elif experiment_file.paired_end == '2':
                pair2[experiment_file.file_loc.split("/")[2]] = experiment_file
            else:
                assert False
    pair1 = sorted(pair1.values())
    pair2 = [pair2[x.paired_with.split("/")[2]] for x in pair1]
    data = (sorted(single_end.values()), pair1, pair2)
    return data

def find_ENCODE_DCC_peaks(experiment_id):
    """Iterate over all bed peak files associated with an experiment.

    Arguments:
    experiment_id: ENCODE experiment to peaks for. 
    Returns: iterator yield ExperimentFile objects
    """
    for experiment_file in find_ENCODE_DCC_experiment_files(experiment_id):
        if ( experiment_file.file_format == 'bed' 
             and experiment_file.output_type in (
                 'peaks', 'optimal idr thresholded peaks')):
            yield experiment_file
    return

def find_ENCODE_DCC_replicate_merged_peaks(experiment_id):
    """Iterate over all bed peak files from merged replicates associated with an experiment.

    Arguments:
    experiment_id: ENCODE experiment to peaks for. 
    Returns: iterator yielding ExperimentFile objects
    """
    for experiment_file in find_ENCODE_DCC_peaks(experiment_id):
        if experiment_file.bsid == 'merged':
            yield experiment_file
    return

def find_ENCODE_DCC_IDR_optimal_peaks(experiment_id):
    """Iterate over all bed 'optimal idr thresholded peaks' files associated with an experiment.

    Arguments:
    experiment_id: ENCODE experiment to peaks for. 
    Returns: iterator yielding ExperimentFile objects
    """
    for experiment_file in find_ENCODE_DCC_replicate_merged_peaks(experiment_id):
        if experiment_file.output_type == 'optimal idr thresholded peaks':
            yield experiment_file
    return

def find_best_ENCODE_DCC_replicate_merged_peak(experiment_id, assembly):
    """Returns a single peak file from merged replicates, prefering optimal IDR peaks when available.

    Arguments:
    experiment_id: ENCODE experiment for which to find peak. 
    Returns: ExperimentFile object, or None if no merged peak files exist
    """
    peak_files = [pk_file for pk_file 
                  in find_ENCODE_DCC_replicate_merged_peaks(experiment_id)
                  if pk_file.assembly == assembly]
    idr_optimal_pk_files = [
        pk_file for pk_file in peak_files 
        if pk_file.output_type == 'optimal idr thresholded peaks']
    if len(idr_optimal_pk_files) > 0:
        if len(idr_optimal_pk_files) > 1:
            warn("Multiple idr optimal peak files for experiment '%s'" % experiment_id)
        return idr_optimal_pk_files[0]
    elif len(idr_optimal_pk_files) == 0 and len(peak_files) > 0:
        if len(peak_files) > 1:
            warn("Multiple merged peak files for experiment '%s'" % experiment_id)
        return peak_files[0]
    return None

################################################################################
#
# Find ENCODE experiment IDs
#
################################################################################
def find_ENCODE_chipseq_experiment_ids(assemblies):
    """Find all ENCODE chipseq experiments from the DCC.  

    """

    # if we pass a string as the assembly, we probably just want that assembly
    # rather than the string iterator
    if isinstance(assemblies, str): assemblies = [assemblies,]
    
    URL = "https://www.encodeproject.org/search/?type=experiment&assay_term_name=ChIP-seq&{}&target.investigated_as=transcription%20factor&limit=all&format=json".format(
        "&".join("assembly=%s"%x for x in assemblies) )
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    for experiment in response_json_dict['@graph']:
        yield experiment['@id'].split("/")[-2]
    return 


def find_ENCODE_DNASE_experiment_ids(assemblies):
    """Find all ENCODE DNASE experiments from the DCC.  

    """
    # if we pass a string as the assembly, we probably just want that assembly
    # rather than the string iterator
    if isinstance(assemblies, str): assemblies = [assemblies,]
    
    URL = "https://www.encodeproject.org/search/?type=experiment&assay_term_name=DNase-seq&{}&limit=all".format(
        "&".join("assembly=%s"%x for x in assemblies) )
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    for experiment in response_json_dict['@graph']:
        yield experiment['@id'].split("/")[-2]
    return 

################################################################################
#
# Download ENCODE Data locally
#
################################################################################

def download_and_index_peak_file(location, output_fname):
    #print "Downloading %i/%i" % (i+1, len(chipseq_exps))
    download_cmd = "wget --quiet {URL}"
    sort_and_compress_cmd = \
        "zcat {FNAME} | sort -k1,1 -k2n -k3n | bgzip -c > {HR_FNAME}"
    mv_cmd = "rm {FNAME}"
    index_cmd = "tabix -p bed {HR_FNAME}"
    
    cmd = " && ".join(
        (download_cmd, sort_and_compress_cmd, mv_cmd, index_cmd)
    ).format(URL=location, 
             FNAME=location.split("/")[-1], 
             HR_FNAME=output_fname )
    return os.system(cmd)

def download_and_index_bam(location, output_fname):
    download_cmd = "wget --quiet {URL} -O {HR_FNAME}"
    index_cmd = "samtools index {HR_FNAME}"
    
    cmd = " && ".join(
        (download_cmd, index_cmd)
    ).format(URL=location, 
             FNAME=location.split("/")[-1], 
             HR_FNAME=output_fname )
    return os.system(cmd)
    

def download_sort_and_index_tfs():
    res = []
    for target, files in find_peaks_and_group_by_target():
        tf_label, tf_name, uniprot_ids = find_target_info(target)
        all_genes = [ get_ensemble_genes_associated_with_uniprot_id(uid)
                      for uid in uniprot_ids ]
        for file_data in files:
            ofname = file_data.file_loc.split("/")[-1]

            human_readable_ofname = (
                "_".join((tf_label, tf_name, 
                          file_data.output_type, 
                          file_data.sample_type.replace(" ", "-"))) \
                + ".EXP-%s.%s.%s.%s.bgz" % (
                    file_data.exp_id, 
                    file_data.rep_key, 
                    file_data.output_type,
                    file_data.file_type))
            human_readable_ofname = human_readable_ofname.replace(
                "/", "#FWDSLASH#")

            #print "Downloading %i/%i" % (i+1, len(chipseq_exps))
            download_cmd = "wget --quiet {URL}"
            sort_and_compress_cmd = \
               "zcat {FNAME} | sort -k1,1 -k2n -k3n | bgzip -c > {HR_FNAME}"
            mv_cmd = "rm {FNAME}"
            index_cmd = "tabix -p bed {HR_FNAME}"
            #print cmd
            if "/" in human_readable_ofname:
                cmd = " ; ".join(
                    (download_cmd, sort_and_compress_cmd, mv_cmd, index_cmd)
                ).format(URL=BASE_URL+file_data.file_loc, 
                         FNAME=ofname, 
                         HR_FNAME=human_readable_ofname )
                print( cmd )
                os.system(cmd)
                res.append(
                    (file_data.exp_id, tf_name, tf_label, 
                     ",".join(uniprot_ids), 
                     ",".join(",".join(genes) for genes in all_genes), 
                     file_data.sample_type, 
                     file_data.output_type,
                     BASE_URL[:-1]+file_data.file_loc, 
                     human_readable_ofname))

def download_and_index_bams():
    res = []
    for target, files in find_peaks_and_group_by_target():
        tf_label, tf_name, uniprot_ids = find_target_info(target)
        all_genes = [ get_ensemble_genes_associated_with_uniprot_id(uid)
                      for uid in uniprot_ids ]
        for file_data in files:
            print( file_data )
            continue
            ofname = file_data.file_loc.split("/")[-1]

            human_readable_ofname = (
                "_".join((tf_label, tf_name, 
                          file_data.output_type, 
                          file_data.sample_type.replace(" ", "-"))) \
                + ".EXP-%s.%s.%s.%s.bgz" % (
                    file_data.exp_id, 
                    file_data.rep_key, 
                    file_data.output_type,
                    file_data.file_type))
            human_readable_ofname = human_readable_ofname.replace(
                "/", "#FWDSLASH#")

            #print "Downloading %i/%i" % (i+1, len(chipseq_exps))
            download_cmd = "wget --quiet {URL}"
            sort_and_compress_cmd = \
               "zcat {FNAME} | sort -k1,1 -k2n -k3n | bgzip -c > {HR_FNAME}"
            mv_cmd = "rm {FNAME}"
            index_cmd = "tabix -p bed {HR_FNAME}"
            #print cmd
            if "/" in human_readable_ofname:
                cmd = " ; ".join(
                    (download_cmd, sort_and_compress_cmd, mv_cmd, index_cmd)
                ).format(URL=BASE_URL+file_data.file_loc, 
                         FNAME=ofname, 
                         HR_FNAME=human_readable_ofname )
                print( cmd )
                os.system(cmd)
                res.append(
                    (file_data.exp_id, tf_name, tf_label, 
                     ",".join(uniprot_ids), 
                     ",".join(",".join(genes) for genes in all_genes), 
                     file_data.sample_type, 
                     file_data.output_type,
                     BASE_URL[:-1]+file_data.file_loc, 
                     human_readable_ofname))

def add_chipseq_controls():
    from DB import conn
    cur = conn.cursor()
    query = "select distinct encode_experiment_id from encode_chipseq_bam_files NATURAL JOIN encode_chipseq_experiments where control is NULL and target not IN (select chipseq_target_id from chipseq_targets where tf_name = 'Control');"
    cur.execute(query, [])
    experiment_ids = [ x[0] for x in cur.fetchall()]
    for exp_index, experiment_id in enumerate(experiment_ids):
        print exp_index, len(experiment_ids), experiment_id
        for control in find_ENCODE_DCC_experiment_controls(experiment_id):
            try: files_data = list(find_ENCODE_DCC_bams(control))
            except: continue
            for file_data in files_data:
                if file_data.assembly != 'hg19': continue
                control_exp_id = find_or_insert_chipseq_experiment(
                    control, file_data.target_id, file_data.sample_type)
                print "Control Exp ID:", control_exp_id
                # insert the control file into the DB
                try: 
                    control_key = insert_chipseq_bam_into_db(file_data)
                except Exception, inst:
                    query = """
                    SELECT encode_chipseq_bam_key 
                      FROM encode_chipseq_bam_files
                     WHERE remote_filename = %s;
                    """
                    cur.execute(query, [BASE_URL[:-1]+file_data.file_loc,])
                    res = cur.fetchall()
                    print "Matching bams", res
                    assert len(res) == 1
                    assert len(res[0]) == 1
                    control_key = res[0][0]
                # find the matching experiment file
                query = """
                SELECT encode_chipseq_bam_key, control
                  FROM encode_chipseq_bam_files, genomes
                 WHERE encode_chipseq_bam_files.annotation = genomes.annotation_id
                   AND encode_experiment_id = %s
                   AND bsid = %s
                   AND rep_key = %s
                   AND genomes.name = %s;
                """
                cur.execute(query, [
                    experiment_id, 
                    file_data.bsid, 
                    str(file_data.rep_key).replace(" ",""), 
                    file_data.assembly
                ])
                #print cur.mogrify(query, [
                #    experiment_id, 
                #    file_data.bsid, 
                #    str(file_data.rep_key).replace(" ",""), 
                #    file_data.assembly
                #])

                res = cur.fetchall()
                print "Bams to update", res, control_key
                if len(res) == 0: continue
                #assert len(res) == 1
                #assert len(res[0]) == 2
                for experiment_file_key, curr_control in res:
                    if curr_control is None:
                        with conn:
                            query = "UPDATE encode_chipseq_bam_files SET control=%s WHERE encode_chipseq_bam_key = %s;"
                            print query
                            cur.execute(query, [control_key, experiment_file_key])
                    else:
                        print "Curr vs New:", curr_control, control_key
                print
        #print list(find_ENCODE_DCC_bams(experiment_id))


def sync_dnase_assays(annotation):
    exp_ids = list(find_ENCODE_chipseq_experiment_ids(annotation)) + list(
        find_ENCODE_DNASE_experiment_ids(annotation))
    for i, exp_id in enumerate(exp_ids):
        files_data = list(find_ENCODE_DCC_experiment_files(exp_id))
        print i, len(exp_ids), exp_id
        for file_data in files_data:
            if file_data.assembly != annotation: continue
            assert file_data.exp_id == exp_id
            exp_id = find_or_insert_ENCODE_experiment(
                exp_id, file_data.sample_type, file_data.assay, file_data.target_id)
            if file_data.file_format == 'bam': 
                try: insert_ENCODE_bam_into_db(file_data)
                except psycopg2.IntegrityError: pass
            elif file_data.output_type in (
                    'peaks', 'optimal idr thresholded peaks'):
                if file_data.file_format == 'bed':
                    insert_ENCODE_peak_file_into_db(file_data)
        
        controls = find_ENCODE_DCC_experiment_controls(exp_id)

def sync_treatments(exp_ids):
    from DB import conn
    cur = conn.cursor()
    query = "INSERT INTO treatments VALUES (%s, %s)"
    for i, exp_id in enumerate(exp_ids):
        _, _, treatments = find_ENCODE_DCC_experiment_metadata(exp_id)
        print i, len(exp_ids), exp_id, treatments
        if treatments is None: continue
        for treatment in treatments:
            cur.execute(query, [exp_id, treatment])
    conn.commit()

def build_fastq_str(all_file_objs):
    rv = [[], [], []]
    for file_objs in all_file_objs:
        for i, data in enumerate(file_objs):
            rv[i].extend(BASE_URL[:-1] + x.file_loc for x in data)
    return ";".join(",".join(x) for x in rv)

def build_anshuls_list(annotation):
    from DB import conn
    cur = conn.cursor()
    query = """
    SELECT tf_name, 
           family_name,
           sample_type, 
           treatments, 
           encode_experiment_id, 
           dnase_encode_experiment_ids
      FROM encode_chipseq_experiments_with_dnase NATURAL JOIN tfs NATURAL JOIN tf_families;
    """
    cur.execute(query, [])
    res = cur.fetchall()
    for i, (tf_name,
         family_name,
         sample_type, 
         treatments, 
         chipseq_exp_id, 
         dnase_encode_experiment_ids) in enumerate(res):
        print >> sys.stderr, i, len(res), chipseq_exp_id
        try:
            #if treatments[0] is None: continue
            #print chipseq_exp_id,
            lab, _, _ = find_ENCODE_DCC_experiment_metadata(chipseq_exp_id)
            input_exp_ids = find_ENCODE_DCC_experiment_controls(chipseq_exp_id)
            #1. TF name -- tfname
            #2. TF family name -- family_name
            #3. Cell type -- sample_type
            #4. Treatment -- dont have
            #5. Lab -- dont have
            #6. comma separated URLs to all ChIP FASTQs
            #7. comma separated URLs to all Input FASTQs
            #8. comma separated URLs to DNase FASTQs
            #9. ChIP sample ENCODE ID -- chipseq_exp_id
            #10. Input sample ENCODE ID -- controls_exp_ids
            #11. DNase sample ENCODE ID -- dnase_exp_id (w/ matching cell_type)
            assert len(treatments) == 1
            treatments_str = (
                'None' if treatments[0] is None 
                else treatments[0].decode('utf-8'))
            chipseq_fastqs_str = build_fastq_str(
                [find_ENCODE_DCC_fastqs(chipseq_exp_id),])        
            input_fastqs_str = build_fastq_str(
                find_ENCODE_DCC_fastqs(exp_id)
                for exp_id in input_exp_ids
            )
            dnase_fastqs_str = build_fastq_str(
                find_ENCODE_DCC_fastqs(exp_id)
                for exp_id in dnase_encode_experiment_ids
            )
            data = [
                tf_name,
                family_name,
                sample_type, 
                treatments_str, 
                lab['title'], 
                chipseq_fastqs_str, 
                input_fastqs_str, 
                dnase_fastqs_str, 
                chipseq_exp_id, 
                ",".join(input_exp_ids), 
                ",".join(dnase_encode_experiment_ids)
            ]
            print u"\t".join(data)
        except:
            continue
        #break

from DB import sync_ENCODE_bam_files
def main():
    annotation = 'hg19'
    #sync_treatments(
    #    list(find_ENCODE_DNASE_experiment_ids(annotation)) + 
    #    list(find_ENCODE_chipseq_experiment_ids(annotation)) 
    #)
    build_anshuls_list(annotation)
    #sync_dnase_assays('annotation')
    #sync_ENCODE_bam_files()

if __name__ == '__main__':
    main()
