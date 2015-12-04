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

from rSeqDNN import init_prediction_script_argument_parser

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
    try:
        for pk, sample, label, score in peaks_and_labels_iterator:
            if peaks_and_labels_iterator._cur_val%10000 == 0:
                print peaks_and_labels_iterator._cur_val, peaks_and_labels_iterator.n
            peaks.append((pk, sample))
            labels.append(label)
            seq = genome_fasta.fetch(pk.contig, pk.start, pk.stop)
            deepbind_process.stdin.write(seq + "\n")
        deepbind_process.stdin.close()
        deepbind_process.wait()
    except Exception as e:
        if e.errno==32:
            print 'DeepBind closed unexpectedly!'
        raise e

    temp_ofp.seek(0)
    for (peak, sample), line, label in izip(peaks, temp_ofp, labels):
        ofp.write("%s_%s_%i_%i\t%i\t%f\n" % (
            sample, peak.contig, peak.start, peak.stop, label, float(line)))
    temp_ofp.close()
    return

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for testing rSeqDNN')
    args = parser.parse_args()

    if args.half_peak_width >= 500:
        raise ValueError('DeepBind requires half peak width less than 500!')
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
            args.skip_ambiguous_peaks)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
        
    return peaks_and_labels, genome_fasta, args.tf_id, args.threads

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
        # XXX Untested 
        pred_labels = np.zeros(len(prbs))
        pred_labels[prbs > 0.5] = 1.0
        pred_labels[prbs <= 0.5] = -1.0
        res = ClassificationResult(labels, pred_labels, prbs)
        results.append(res)
        print "Fold_%i" % fold_i, res
    print results

if __name__ == '__main__':
    main()
