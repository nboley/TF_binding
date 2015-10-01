import os
import argparse

from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import ClassificationResults

from KerasModel import KerasModel

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for training rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')

    parser.add_argument('--tf-id',
                        help='TF to build model on')

    parser.add_argument('--pos-regions', type=getFileHandle,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle,
                        help='regions with negative labels')

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
    
    return peaks_and_labels, args.genome_fasta


def main():
    peaks_and_labels, genome_fasta = parse_args()
    model = KerasModel(peaks_and_labels)
    results = ClassificationResults()
    for train, valid in peaks_and_labels.iter_train_validation_subsets():
        fit_model = model.train_rSeqDNN_model(train, genome_fasta, './test')
        results.append(fit_model.evaluate_rSeqDNN_model(
        encode_peaks_sequence_into_binary_array(
                valid.peaks, genome_fasta), valid.labels))
    print 'Printing cross validation results:'
    print results

if __name__ == '__main__':
    main()
