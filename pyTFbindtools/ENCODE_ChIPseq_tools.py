import os, sys, time

import requests, json

from warnings import warn

import urllib2

from itertools import chain

from collections import defaultdict, namedtuple

import re

import gzip, io

from multiprocessing import Value

ExperimentFile = namedtuple('ExperimentFile', [
    'exp_id', 'assay',
    'target_id', 
    'sample_type', 'rep_key', 'bsid', 
    'assembly',
    'file_format', 'file_format_type', 'output_type', 'file_loc'])

BASE_URL = "https://www.encodeproject.org/"

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
    assert len(treatment_term_names) <= 1
    return ( response_json_dict['lab'], 
             response_json_dict['description'], 
             (next(iter(treatment_term_names)) 
              if len(treatment_term_names) ==1 
              else None)
            )

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
            
        yield ExperimentFile(
            experiment_id, assay,
            target_id, 
            sample_type, rep_key, bsid,
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
    
    URL = "https://www.encodeproject.org/search/?type=experiment&assay_term_name=DNase-seq&{}&limit=all&format=json".format(
        "&".join("assembly=%s"%x for x in assemblies) )
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    for experiment in response_json_dict['@graph']:
        yield experiment['@id'].split("/")[-2]
    return 

################################################################################
#
# Find metadata about ENCODE ChIP-Seq Targets
#
################################################################################

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


def iterate_hg19_chipseq_experiments():
    """This is just an example of iterating through the chipseq experiments.

    """
    for experiment_id in find_ENCODE_DNASE_experiment_ids('hg19'):
        yield experiment_id
    return

def main():
    for experiment_id in iterate_hg19_chipseq_experiments():
        print list(find_ENCODE_DCC_bams(experiment_id))
    

if __name__ == '__main__':
    main()
