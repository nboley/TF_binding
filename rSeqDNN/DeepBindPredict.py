import os
import argparse
import numpy as np
import subprocess
import tempfile
import time

from itertools import izip

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)
from pysam import FastaFile

from grit.lib.multiprocessing_utils import ThreadSafeFile, fork_and_wait

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    PeaksAndLabelsThreadSafeIterator
)

from pyTFbindtools.cross_validation import (
    ClassificationResults, ClassificationResult)

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
    
def get_probability_from_score(score):
    '''pass scores through sigmoid
    '''    
    return 1. / (1. + np.exp(-1.*score))

def evaluate_predictions(probs, y_validation):
    '''Evaluate the quality of deep bind predictions.
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

def load_predictions(fname):
    peaks = []
    prbs = []
    labels = []
    with open(fname) as fp:
        for line in fp:
            peak, label, score = line.split()
            peaks.append(peak)
            prbs.append(get_probability_from_score(float(score)))
            labels.append(int(label))
    return peaks, np.array(prbs), np.array(labels)

def score_regions_with_deepbind(
        peaks_and_labels_iterator, ofp, tf_id, genome_fasta_fname ):
    # make sure the input objects are thread safe
    assert isinstance(
        peaks_and_labels_iterator, PeaksAndLabelsThreadSafeIterator)
    assert isinstance(
        ofp, ThreadSafeFile)
    genome_fasta = FastaFile(genome_fasta_fname)
    
    # get the deep bind model id for this tf_id
    model_id = get_deepbind_model_id(tf_id)

    # get a temporary file to write to
    temp_ofp = tempfile.TemporaryFile()
    
    # spawn a deepbind process
    deepbind_process = subprocess.Popen(
        " ".join(('deepbind', ' --no-head', model_id)),
        stdout=temp_ofp, 
        stdin=subprocess.PIPE,
        shell=True)
    
    # write the peaks sequence to a fasta file
    peaks = []
    labels = []
    for pk, sample, label in peaks_and_labels_iterator:
        if peaks_and_labels_iterator._cur_val%10000 == 0:
            print peaks_and_labels_iterator._cur_val, peaks_and_labels_iterator.n
        peaks.append((pk, sample))
        labels.append(label)
        seq = genome_fasta.fetch(pk.contig, pk.start, pk.stop)
        deepbind_process.stdin.write(seq + "\n")
    deepbind_process.stdin.close()
    deepbind_process.wait()
    
    temp_ofp.seek(0)
    for (peak, sample), line, label in izip(peaks, temp_ofp, labels):
        ofp.write("%s_%s_%i_%i\t%i\t%f\n" % (
            sample, peak.contig, peak.start, peak.stop, label, float(line)))
    temp_ofp.close()
    return

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
    parser.add_argument('--half-peak-width', type=int, default=400,
                        help='half peak width about summits for training')

    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to use')

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
    
    return peaks_and_labels, args.genome_fasta, args.tf_id, args.threads

def main():
    peaks_and_labels, genome_fasta, tf_id, num_threads = parse_args()
    results = ClassificationResults()
    for fold_i, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets()):
        peaks_and_labels_iterator = valid.thread_safe_iter()
        ofname = "scores.%s.fold%i.txt" % (tf_id, fold_i)
        ofp = ThreadSafeFile(ofname, "w")
        args = [peaks_and_labels_iterator, ofp, tf_id,genome_fasta.filename]
        fork_and_wait(num_threads, score_regions_with_deepbind, args)
        ofp.close()
        peaks, prbs, labels = load_predictions(ofname)
        res = evaluate_predictions(prbs, labels)
        print "Fold_%i" % fold_i, res


if __name__ == '__main__':
    main()
