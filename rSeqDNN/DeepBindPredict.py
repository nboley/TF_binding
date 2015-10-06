import os
import argparse
import numpy as np
import subprocess
import tempfile

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

def get_deepbind_model_id(tf_id):
    if tf_id=='T014210_1.02': # MYC
        return 'D00785.001'
    elif tf_id=='T011266_1.02': # MAX
        return 'D00504.005'
    elif tf_id=='T044261_1.02': # YY1
        return 'D00710.007'
    elif tf_id=='T044268_1.02': # CTCF
        return 'D00328.018'
    else:
        raise ValueError('this TF is not supported for DeepBind predictions!')

def subprocess_cmd(command):
    '''runs shell command and return output
    '''
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    
    return proc_stdout

def score_seq_with_deepbind_model(model_id, input_fasta):
    '''scores sequences and returns arrays with scores
    TODO: check deepbind is in PATH
    '''
    command = ' '.join(['deepbind', ' --no-head', model_id, input_fasta])
    # open tempfile, store command results there
    with tempfile.TemporaryFile() as tf:
        process = subprocess.Popen(command,stdout=tf, shell=True)
        process.wait() # wait for it to finish
        tf.flush()
        tf.seek(0)
        data = np.asarray(tf.read().rstrip().split("\n"), dtype='float')

    return data
    
def get_probabilites_from_scores(scores):
    '''pass scores through sigmoid
    '''
    
    return 1. / (1. + np.exp(-1.*scores))

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
    parser.add_argument('--half-peak-width', type=int, default=400,
                        help='half peak width about summits for training')

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
    
    return peaks_and_labels, args.genome_fasta, args.output_fasta_filename_prefix, args.tf_id

import time

def main():
    peaks_and_labels, genome_fasta, fasta_filename_prefix, tf_id = parse_args()
    results = ClassificationResults()
    for train, valid in peaks_and_labels.iter_train_validation_subsets():
        print 'writing sequences and labels to files..'
        for sample in valid.sample_ids:
            for contig in valid.contigs:
                subset_fasta_filename = '_'.join([fasta_filename_prefix, sample, contig]) + '.fa'
                print subset_fasta_filename
                subset_labels_filename = '.'.join([subset_fasta_filename, 'labels'])
                subset_validation_data = valid.subset_data([sample], [contig])
                encode_peaks_sequence_into_fasta_file(subset_validation_data,
                                                      genome_fasta,
                                                      subset_fasta_filename,
                                                      subset_labels_filename)
                model_id = get_deepbind_model_id(tf_id)
                start_time = time.time()
                scores = score_seq_with_deepbind_model(model_id, subset_fasta_filename)
                end_time = time.time()
                print 'num of sequences: ', len(scores)
                print 'time per sequence scoring: ', 1.*(end_time-start_time)/len(scores)
                probabilities = get_probabilites_from_scores(scores)
                result = evaluate_predictions(probabilities,
                                              subset_validation_data.labels)
                print 'result: '
                print result
                break
            break
        print 'done!'
        break


if __name__ == '__main__':
    main()
