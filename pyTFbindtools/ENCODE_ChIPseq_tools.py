import os, sys, time

import requests, json

import urllib

from itertools import chain

from collections import defaultdict, namedtuple

import re

import gzip, io

from multiprocessing import Value

PeakFile = namedtuple('PeakFiles', [
    'exp_id', 'target_id', 'sample_type', 'rep_key', 'bsid', 
    'file_type', 'output_type', 'file_loc'])

BAMFileMetaDataNamedTuple = namedtuple('BAMFileMetaData', [
    'exp_id', 'target_id', 'sample_type', 'rep_key', 'bsid', 
    'file_type', 'output_type', 'file_loc'])
class BAMFileMetaData(BAMFileMetaDataNamedTuple):
    """
    ChipSeq.GM12878.BCL11A.BS1.ENCFF000NSQ.bam.bai
    """
    def build_fname(self):
        fields = self._asdict()
        fields['rep_desc'] = "_".join(str(x) for x in self.rep_key)
        fields['target_desc'] = self.target_id.split("/")[-2].split("-")[0]
        return "ChIPSeq.{sample_type}.{target_desc}.BSID-{bsid}.REPID-{rep_desc}.EXPID-{exp_id}.bam".format(**fields)

BASE_URL = "https://www.encodeproject.org/"

def get_TF_name_and_label_from_fname(fname):
    TF_label = os.path.basename(fname).split("_")[0]
    TF_name = os.path.basename(fname).split("_")[1]
    return TF_name, TF_label

def build_TF_fname(exp_id, sample_type, re):
    TF_label = os.path.basename(fname).split("_")[0]
    TF_name = os.path.basename(fname).split("_")[1]
    return TF_name, TF_label

def find_uniformly_processed_called_peaks(experiment_id, only_merged=True):
    URL = "https://www.encodeproject.org/experiments/{}/".format(experiment_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()

    target_id = response_json_dict['target']['@id']
    target_label = response_json_dict['target']['label']

    # build the replicate mapping
    replicates = {}
    for rep_rec in response_json_dict['replicates']:
        key = (rep_rec['biological_replicate_number'], 
               rep_rec['technical_replicate_number'])
        replicates[key] = rep_rec

    sample_type = response_json_dict['biosample_term_name']

    for file_rec in response_json_dict['files']:
        file_type = file_rec['file_format']
        output_type = file_rec['output_type']
        # the uniformily processed peaks are this type
        if file_type not in ['bed_broadPeak', 'bed_narrowPeak']:
            continue
        
        if 'replicate' not in file_rec: 
            rep_key = 'merged'
            bsid = 'merged'
        else:
            if only_merged: continue
            rep_key = (file_rec['replicate']['biological_replicate_number'],
                       file_rec['replicate']['technical_replicate_number'] )
            bsid = replicates[rep_key]['library']['biosample']['accession']
        file_loc = file_rec['href']
        yield PeakFile(experiment_id, target_id, sample_type, rep_key, bsid, 
                       file_type, output_type, file_loc)

    return 

def find_bams(experiment_id):
    URL = "https://www.encodeproject.org/experiments/{}/".format(experiment_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()

    target_id = response_json_dict['target']['@id']
    target_label = response_json_dict['target']['label']

    # build the replicate mapping
    replicates = {}
    for rep_rec in response_json_dict['replicates']:
        key = (rep_rec['biological_replicate_number'], 
               rep_rec['technical_replicate_number'])
        replicates[key] = rep_rec

    sample_type = response_json_dict['biosample_term_name']

    for file_rec in response_json_dict['files']:
        file_type = file_rec['file_format']
        output_type = file_rec['output_type']
        if file_type not in ['bam', ]:
            continue
        
        if 'replicate' not in file_rec: 
            rep_key = 'merged'
            bsid = 'merged'
        else:
            rep_key = (file_rec['replicate']['biological_replicate_number'],
                       file_rec['replicate']['technical_replicate_number'] )
            bsid = replicates[rep_key]['library']['biosample']['accession']
        file_loc = BASE_URL + file_rec['href']
        yield BAMFileMetaData(
            experiment_id, target_id, sample_type, rep_key, bsid, 
            file_type, output_type, file_loc)
    return 

def find_chipseq_experiments(assemblies=['mm9', 'hg19']): # 'hg19', 
    if isinstance(assemblies, str):
        assemblies = [assemblies,]
    URL = "https://www.encodeproject.org/search/?type=experiment&assay_term_name=ChIP-seq&{}&target.investigated_as=transcription%20factor&limit=all&format=json".format(
        "&".join("assembly=%s"%x for x in assemblies) )
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    biosamples = set()
    for experiment in response_json_dict['@graph']:
        yield experiment['@id'].split("/")[-2], experiment
    return 

def find_target_info(target_id):
    URL = "https://www.encodeproject.org/{}?format=json".format(target_id)
    response = requests.get(URL, headers={'accept': 'application/json'})
    response_json_dict = response.json()
    uniprot_ids = [x[10:] for x in response_json_dict['dbxref']
                   if x.startswith("UniProtKB:")]
    return response_json_dict['label'], response_json_dict['name'], uniprot_ids

def get_ensemble_genes_associated_with_uniprot_id(uniprot_id):
    ens_id_pat = '<property type="gene ID" value="(ENS.*?)"/>'
    res = urllib2.urlopen("http://www.uniprot.org/uniprot/%s.xml" % uniprot_id)
    data = res.read()
    gene_ids = set(re.findall(ens_id_pat, data))
    return sorted(gene_ids)

def find_peaks_and_group_by_target(
        only_merged=True, prefer_uniformly_processed=True):
    # find all chipseq experiments and group them by target
    targets = defaultdict(list)
    chipseq_exps = list(find_chipseq_experiments())
    for i, exp_id in enumerate(chipseq_exps):
        #if i > 10: break
        print( i, len(chipseq_exps), exp_id )
        for rep_i, res in enumerate(find_called_peaks(exp_id, True)):
            targets[res.target_id].append(res)
            #print i, find_target_info(res.target_id)

    # remove redudant experiments
    for i, (target, file_data) in enumerate(targets.items()):
        any_uniformly_processed = any(
            f.output_type == 'UniformlyProcessedPeakCalls'
            for f in file_data )

        new_file_data = [
            f for f in file_data 
            if (not prefer_uniformly_processed 
                or not any_uniformly_processed 
                or f.output_type=='UniformlyProcessedPeakCalls')
            and (
                    not only_merged
                    or f.bsid == 'merged'
            )
        ]
        print( i, len(targets), target )
        yield target, new_file_data
    return

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

if __name__ == '__main__':
    experiments = find_chipseq_experiments('hg19')
    for exp_id, exp_meta_data in experiments:
        biosample_type = exp_meta_data['biosample_term_name']
        TF = exp_meta_data['target']['label']
        if biosample_type != 'GM12878': continue
        print exp_id, biosample_type, TF
        #for data in find_bams(exp):
        #    print data
