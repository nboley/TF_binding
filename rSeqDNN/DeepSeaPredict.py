import os
import argparse
import numpy as np
import subprocess
import tempfile
import time
from math import log
from distutils.util import strtobool

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)
from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import (
    ClassificationResults, ClassificationResult)


def encode_peaks_sequence_into_fasta_file(validation_data, fasta, fasta_ofname, labels_ofname):
    '''writes data peaks sequence into file and saves labels
    '''
    np.savetxt(labels_ofname, validation_data.labels)
    peaks = validation_data.peaks
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    with open(fasta_ofname, 'w') as wf:
        for i, pk in enumerate(peaks):
            seq = fasta.fetch(pk.contig, pk.start, pk.stop)
            wf.write('>'+str(i)+'\n')
            wf.write(seq+'\n')

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
        raise ValueError('this sample is not supported for DeepSea predictions!')

def get_deepsea_tf_name(tf_id):
    if tf_id=='T014210_1.02': # MYC
        return 'c-Myc'
    elif tf_id=='T011266_1.02': # MAX
        return 'MAX'
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
    columns = [int(i) for i, h in enumerate(header_list) if sample_name==h.split('|')[0].strip() and tf_name==h.split('|')[1]]
    if len(columns) > 1: # notify user if there's more than one match
        print 'found more than matching task for', tf_name, sample_name
        for column in columns:
            print header_list[column]
    scores = np.asarray(read_array[:, columns[0]],
                        dtype='float') # default to first match
    
    return scores

def score_seq_with_deepbind_model(tf_id, sample, input_fasta, output_dir):
    '''scores sequences and returns arrays with scores
    TODO: check rundeepsea.py is in PATH
    '''
    command = ' '.join(['python ./rundeepsea_fixed.py',
                        input_fasta,
                        output_dir])
    # open tempfile, store command results there
    with tempfile.NamedTemporaryFile() as tf:
        process = subprocess.Popen(command,stdout=tf, shell=True)
        process.wait() # wait for it to finish
        tf.flush()
        tf.seek(0)
    wd = os.getcwd()
    output_filename = '/'.join([wd, output_dir, 'infile.fasta.out'])
    scores = get_scores_from_deepsea_output(open(output_filename).readlines(),
                                            tf_id,
                                            sample)
   
    return scores
    
def get_probabilites_from_scores(scores):
    '''pass scores through sigmoid
    '''
    
    return 1. / (1. + np.exp(-1.*scores))

def normalize_deepsea_scores(scores, tf_id, sample):
    '''
    normalizes raw deepsea scores based on formula provided in
    http://deepsea.princeton.edu/help/
    '''
    tf_name = get_deepsea_tf_name(tf_id) # get deepsea tf name
    sample_name = get_deepsea_sample_name(sample) # get deepsea sample name
    full_table = np.loadtxt(
        './posproportion.txt', skiprows=1, delimiter='\t', dtype='str')
    matching_sample_indices = full_table[:, 1] == sample_name
    matching_sample_table =  full_table[matching_sample_indices, :]
    matching_sample_tf_indices = matching_sample_table[:, 2] == tf_name
    matching_sample_tf_table = matching_sample_table[
        matching_sample_tf_indices, :]
    c_train = float(matching_sample_tf_table[0, -1]) # default to first match
    # evaluate 1/(1+exp(-( log(P/(1-P))+log(5%/(1-5%))-log(c_train/(1-c_train ))) ))
    normalized_scores = 1/(1+np.exp(-(np.log(np.divide(scores, 1-scores))+log(0.05/0.95)-log(c_train/(1.-c_train))))) 
    
    return normalized_scores

def evaluate_predictions(probs, y_validation):
    '''
    '''
    preds = np.asarray(probs > 0.5, dtype='int')
    true_pos = y_validation == 1
    true_neg = y_validation == 0
    precision, recall, _ = precision_recall_curve(y_validation, probs)
    prc = np.array([recall,precision])
    auPRC = auc(recall, precision)
    auROC = roc_auc_score(y_validation, probs)
    classification_result = ClassificationResult(
        None, None, None, None, None, None,
        auROC, auPRC,
        np.sum(preds[true_pos] == 1), np.sum(true_pos),
        np.sum(preds[true_neg] == 0), np.sum(true_neg)
    )

    return classification_result

def download_and_fix_deepsea():
    if not os.path.exists("./DeepSEA-v0.93/rundeepsea_fixed.py"):
        # download deepsea
        download_deepsea_script = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "./download_deepsea.sh"))
        if not os.path.exists(download_deepsea_script):
            raise ValueError('download_deepsea.sh is missing! exiting!')
        process = subprocess.Popen(
            'bash %s' % download_deepsea_script, shell=True)
        process.wait() # wait for it to finish
        deepsea_lines = open('./DeepSEA-v0.93/rundeepsea.py', 'r').readlines()
        with open('./DeepSEA-v0.93/rundeepsea_fixed.py', 'w') as wf:
            wf.write('#!/usr/bin/env python') # modify 
            for line in deepsea_lines[:16]:
                wf.write(line)
            # fix their bug
            wf.write("outfilename = outdir + infilename.split(\'/\')[-1]+\'.out\'")
            for line in deepsea_lines[16:]:
                wf.write(line)

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for testing rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')

    parser.add_argument('--tf-id', required=True,
                        help='TF to build model on')

    parser.add_argument('--pos-regions', type=getFileHandle,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle,
                        help='regions with negative labels')
    parser.add_argument('--output-fasta-filename-prefix', required=True,
                        help='prefix for fasta files with test sequences')
    parser.add_argument('--half-peak-width', type=int, default=500,
                        help='half peak width about summits for training')
    parser.add_argument('--output-directory', required=True,
                        help='output directory for deepsea results')
    parser.add_argument('--normalize', type=strtobool, default=True,
                        help='normalize deepsea scores')
    parser.add_argument( '--max-num-peaks-per-sample', type=int, 
        help='the maximum number of peaks to parse for each sample (used for debugging)')

    args = parser.parse_args()
    
    if args.tf_id != None:
        assert args.pos_regions == None and args.neg_regions == None, \
            "It doesnt make sense to set both tf-id and either --pos-regions or --neg-regions"
        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            args.tf_id, args.half_peak_width, args.max_num_peaks_per_sample)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
    
    return peaks_and_labels, args.genome_fasta, args.output_fasta_filename_prefix, args.tf_id, args.output_directory, args.normalize

def main():
    download_and_fix_deepsea()
    peaks_and_labels, genome_fasta, fasta_filename_prefix, tf_id, output_directory, normalize = parse_args()
    results = ClassificationResults()
    cwd = os.getcwd()
    os.chdir('./DeepSEA-v0.93/') # cd to run deepsea
    for train, valid in peaks_and_labels.iter_train_validation_subsets():
        print 'writing sequences and labels to files..'
        for sample in valid.sample_ids:
            for contig in valid.contigs:
                # name fasta file, has to end with .fasta for deepsea to read it - facepalm.
                subset_fasta_filename = '_'.join([fasta_filename_prefix, sample, contig]) + '.fasta'
                print subset_fasta_filename
                subset_labels_filename = '.'.join([subset_fasta_filename, 'labels'])
                subset_validation_data = valid.subset_data([sample], [contig])
                encode_peaks_sequence_into_fasta_file(subset_validation_data,
                                                      genome_fasta,
                                                      subset_fasta_filename,
                                                      subset_labels_filename)
                start_time = time.time()
                scores = score_seq_with_deepbind_model(tf_id, sample,
                                                       subset_fasta_filename,
                                                       output_directory)
                end_time = time.time()
                os.chdir(cwd) # cd back
                if normalize:
                    print 'normalizing scores...'
                    scores = normalize_deepsea_scores(scores, tf_id, sample)
                print 'num of sequences: ', len(scores)
                print 'time per sequence scoring: ', 1.*(end_time-start_time)/len(scores)
                result = evaluate_predictions(scores,
                                              subset_validation_data.labels)
                print 'result: '
                print result
                break
            break
        print 'done!'
        break


if __name__ == '__main__':
    main()
