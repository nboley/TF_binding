import os
import argparse
import numpy as np
import subprocess
import tempfile
import time  
import multiprocessing
from math import log
from joblib import Parallel, delayed  

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)
from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    PeaksAndLabelsThreadSafeIterator
)
from pyTFbindtools.cross_validation import ClassificationResult
from rSeqDNN import init_prediction_script_argument_parser

from grit.lib.multiprocessing_utils import ThreadSafeFile, fork_and_wait

def encode_peaks_sequence_into_fasta_file(
        validation_data_iterator, fasta, fasta_tsf):
    '''writes data peaks sequence into file
    '''
    labels = []
    for i, (pk, _, label, _) in enumerate(validation_data_iterator):
        seq = fasta.fetch(pk.contig, pk.start, pk.stop)
        # if the sequence is not the same size as the peak,
        # it must run off the chromosome so skip it
        if len(seq) != pk.pk_width: 
            continue
        if label == 0: # skip ambiguous peaks if included
            continue
        fasta_tsf.write('>'+str(i)+'\n')
        fasta_tsf.write(seq+'\n')
        labels.append(label)
    
    return np.array(labels) 

def get_deepsea_sample_name(sample):
    if sample=='E003':
        return 'H1-hESC'
    elif sample=='E114':
        return 'A549'
    elif sample=='E116':
        return 'GM12878'
    elif sample=='E117':
        return 'HeLa-S3'
    elif sample=='E118':
        return 'HepG2'
    elif sample=='E119':
        return 'HMEC'
    elif sample=='E120':
        return 'HSMM'
    elif sample=='E121':
        return 'HSMMtube'
    elif sample=='E122':
        return 'HUVEC'
    elif sample=='E123':
        return 'K562'
    elif sample=='E124':
        return 'Monocytes-CD14+RO01746'
    elif sample=='E125':
        return 'NH-A'
    elif sample=='E126':
        return 'NHDF-Ad'
    elif sample=='E127':
        return 'NHEK'
    elif sample=='E128':
        return 'NHLF'
    elif sample=='E129':
        return 'Osteoblasts'
    else:
        raise ValueError(
            'this sample is not supported for DeepSea predictions!')

def get_deepsea_tf_name(tf_id):
    if tf_id=='T014210_1.02': # MYC
        return 'c-Myc'
    elif tf_id=='T011266_1.02': # MAX
        return 'Max'
    elif tf_id=='T044261_1.02': # YY1
        return 'YY1'
    elif tf_id=='T044268_1.02': # CTCF
        return 'CTCF'
    else:
        raise ValueError('this TF is not supported for DeepSea predictions!')

def get_scores_from_deepsea_output(read_lines, tf_id, sample):
    '''get deepsea probabilities for tf_name from deepsea output
    '''
    num_examples = len(read_lines)
    num_tasks = len(read_lines[0].split(','))
    read_data = []
    for read_line in read_lines[1:]: # put output in array
        read_data.append(read_line.split(','))
    read_array = np.asarray(read_data)
    header_line = read_lines[0] # deepsea header with task names
    header_list = header_line.split(',')
    tf_name = get_deepsea_tf_name(tf_id) # get deepsea tf name
    sample_name = get_deepsea_sample_name(sample) # get deepsea sample name
    # parse deepsea header and find task with matching tf and sample name
    columns = [ int(i) for i, h in enumerate(header_list) 
                if sample_name==h.split('|')[0].strip() 
                and tf_name==h.split('|')[1] ]
    if len(columns) > 1: # notify user if there's more than one match
        print 'found more than matching task for', tf_name, sample_name
        for column in columns:
            print header_list[column]
    # default to first match
    scores = np.asarray(read_array[:, columns[0]], dtype='float')
    
    return scores

def score_seq_with_deepsea_model(tf_id, sample, input_fasta, output_dir):
    '''scores sequences and returns arrays with scores
    TODO: check rundeepsea.py is in PATH
    '''
    command = ' '.join(['python ./rundeepsea_fixed.py',
                        input_fasta,
                        output_dir])
    # run command
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    process.wait() # wait for it to finish
    wd = os.getcwd()
    output_filename = '/'.join([wd, output_dir, 'infile.fasta.out'])
    scores = get_scores_from_deepsea_output(open(output_filename).readlines(),
                                            tf_id,
                                            sample)
   
    return scores

def normalize_deepsea_scores(scores, tf_id, sample):
    '''
    normalizes raw deepsea scores based on formula provided in
    http://deepsea.princeton.edu/help/
    '''
    tf_name = get_deepsea_tf_name(tf_id)
    sample_name = get_deepsea_sample_name(sample)
    # deepsea score normalization table
    # XXX maybe rename this file to something more descriptive
    full_table = np.loadtxt(
        './posproportion.txt', skiprows=1, delimiter='\t', dtype='str')
    matching_sample_indices = (full_table[:, 1] == sample_name)
    matching_sample_table = full_table[matching_sample_indices, :]
    matching_sample_tf_indices = (matching_sample_table[:, 2] == tf_name)
    matching_sample_tf_table = matching_sample_table[
        matching_sample_tf_indices, :]
    # default to first match
    # XXX print out a warning if there's more than 1 match
    c_train = float(matching_sample_tf_table[0, -1]) 
    ## Formula is taken from the deepsea website (
    ##     http://deepsea.princeton.edu/help/ bullet point 6) 
    # evaluate 1/(1+exp(-( log(P/(1-P))+log(5%/(1-5%))-log(c_train/(1-c_train ))) ))
    # normalize by the percent positives in the training set (?) says the website....
    alpha = 0.05
    normalized_scores = 1/(1+np.exp(-(np.log(np.divide(scores, 1-scores))
                                      +log(alpha/(1-alpha))
                                      -log(c_train/(1.-c_train))
                                   ))) 
    
    return normalized_scores

def download_and_fix_deepsea():
    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    DEEPSEA_PATH = os.path.join(BASE_PATH, "./DeepSEA-v0.93/")
    deepsea_script = os.path.join(
        DEEPSEA_PATH, "rundeepsea_fixed.py")
    patched_deepsea_script = os.path.join(
        DEEPSEA_PATH, "rundeepsea.py")
    download_deepsea_script = os.path.join(
        BASE_PATH, "./download_deepsea.sh")
    # If we've already done this, then continue
    if os.path.exists(patched_deepsea_script):
        return

    initial_wd = os.getcwd()

    # download deepsea
    if not os.path.exists(download_deepsea_script):
        os.chdir(BASE_PATH) # cd to run deepsea
        if not os.path.exists(download_deepsea_script):
            raise ValueError('download_deepsea.sh is missing! exiting!')
        process = subprocess.Popen(
            'bash %s' % download_deepsea_script, shell=True)
        process.wait() 

    ## XXX In the future use the UNIX patch command
    os.chdir(DEEPSEA_PATH) # cd to run deepsea    
    with open(deepsea_script, 'r') as fp:
        deepsea_lines = fp.readlines()
    with open(patched_deepsea_script, 'w') as wf:
        wf.write('#!/usr/bin/env python') # modify 
        for line in deepsea_lines[:16]:
            wf.write(line)
        # fix their bug
        wf.write("outfilename = outdir + infilename.split(\'/\')[-1]+\'.out\'")
        for line in deepsea_lines[16:]:
            wf.write(line)
    
    os.chdir(initial_wd)
    
    return

def load_results(fname):
    '''load results from deepsea run
OB    '''
    labels = []
    pred_labels = []
    scores = []
    with open(fname) as fp:
        for line in fp:
            label, pred_label, score = line.split()
            labels.append(float(label))
            pred_labels.append(float(pred_label))
            scores.append(float(score))
            
    return np.array(labels), np.array(pred_labels), np.array(scores)

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for testing rSeqDNN')

    parser.add_argument('--output-fasta-filename-prefix', required=True,
                        help='prefix for fasta files with test sequences')
    parser.add_argument('--output-directory', required=True,
                        help='output directory for deepsea results')
    parser.add_argument('--normalize', default=True,
                        help='normalize deepsea scores')

    args = parser.parse_args()

    assert args.annotation_id is not None or args.genome_fasta is not None, \
        "Must set either --annotation-id or --genome-fasta"
    if args.genome_fasta is None:
        assert args.annotation_id is not None
        from pyTFbindtools.DB import load_genome_metadata
        genome_fasta = FastaFile(
            load_genome_metadata(args.annotation_id).filename) 
    else:
        genome_fasta = args.genome_fasta
    
    if args.tf_id is not None:
        assert args.pos_regions is None and args.neg_regions is None, \
            "It doesnt make sense to set both --tf-id and either --pos-regions or --neg-regions"
        assert args.annotation_id is not None, \
            "--annotation-id must be set if --tf-id is set"
        assert args.genome_fasta is None, \
            "if --tf-id is set the genome fasta must be specified by the --annotation-id"

        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            args.tf_id,
            args.annotation_id,
            args.half_peak_width, 
            args.max_num_peaks_per_sample, 
            args.include_ambiguous_peaks)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
    
    return ( peaks_and_labels, 
             genome_fasta, 
             args.output_fasta_filename_prefix, 
             args.tf_id, 
             args.output_directory, 
             args.normalize,
             args.threads)

def run_deepsea(input_list):
    '''
    runs all functions
    '''
    print 'starting deepsea run..'
    ( validation_data_iterator, 
      sample,
      contigs,
      normalize, 
      fasta_filename_prefix, 
      genome_fasta_fname, 
      tf_id, 
      output_directory,
      ofp ) = input_list
    # check objects are thread safe
    assert isinstance(
        validation_data_iterator, PeaksAndLabelsThreadSafeIterator)
    assert isinstance(
        ofp, ThreadSafeFile)
    genome_fasta = FastaFile(genome_fasta_fname)
    
    initial_wd = os.getcwd()
    os.chdir('./DeepSEA-v0.93/') # cd to run deepsea
    # name fasta file, has to end with .fasta for deepsea to read it - facepalm.
    pid = str(int(os.getpid()))
    subset_fasta_filename = "%s_%s_%s.fasta" % (
        fasta_filename_prefix, sample, pid)
    fasta_tsf = ThreadSafeFile(subset_fasta_filename, "w")

    labels = encode_peaks_sequence_into_fasta_file(
        validation_data_iterator,
        genome_fasta,
        fasta_tsf)
    fasta_tsf.close()
    subset_output_directory = output_directory + pid
    start_time = time.time()
    scores = score_seq_with_deepsea_model(
        tf_id, sample, subset_fasta_filename, subset_output_directory)
    end_time = time.time()
    
    os.chdir(initial_wd)
    if normalize:
        print 'normalizing scores...'
        scores = normalize_deepsea_scores(scores, tf_id, sample)
        print 'num of sequences: ', len(scores)
        print 'time per sequence scoring: ', float(
            end_time-start_time)/len(scores)
    
    pred_labels = np.zeros(len(scores))
    pred_labels[scores > 0.5] = 1.0
    pred_labels[scores <= 0.5] = -1.0
    assert len(labels)==len(pred_labels)
    assert len(pred_labels)==len(scores)
    for i in xrange(len(scores)):
        ofp.write("%f %f %f\n" % (labels[i],
                                  pred_labels[i],
                                  scores[i])) 
    
def main():
    download_and_fix_deepsea()
    ( peaks_and_labels, 
      genome_fasta, 
      fasta_filename_prefix, 
      tf_id, 
      output_directory, 
      normalize,
      num_threads ) = parse_args()
    results = []
    validation_contigs = ['chr8', 'chr9']
    for sample in peaks_and_labels.sample_ids:
        print 'subsetting to deepsea test data...'
        validation_data = peaks_and_labels.subset_data([sample],
                                                       validation_contigs)
        print 'creating thread safe iterator..'
        validation_data_iterator = validation_data.thread_safe_iter()
        ofname = "deepsea.results.%s.%s.txt" % (tf_id, sample)
        ofp = ThreadSafeFile(ofname, "w")
        inputs = [[validation_data_iterator, sample, validation_contigs,
                  normalize, fasta_filename_prefix, genome_fasta.filename,
                  tf_id, output_directory, ofp]]
        print 'spawning deepsea run...'
        fork_and_wait(num_threads, run_deepsea, inputs)
        ofp.close()
        labels, pred_labels, scores = load_results(ofname)
        result = ClassificationResult(labels, pred_labels, scores)
        results.append(result)
    print 'printing results from all test runs...'
    for i, sample in enumerate(peaks_and_labels.sample_ids):
        print 'sample: ', sample
        print results[i]
   
if __name__ == '__main__':
    main()
