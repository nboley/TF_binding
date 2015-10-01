import os
import argparse

from pysam import FastaFile

from pyTFbindtools.peaks import load_labeled_peaks_from_beds, getFileHandle

from pyTFbindtools.cross_validation import ClassificationResults

from KerasModel import KerasModel

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for training rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')
    parser.add_argument('--pos-regions', type=getFileHandle, required=True,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle, required=True,
                        help='regions with negative labels')
    parser.add_argument('--half-peak-width', type=int, default=400,
                        help='half peak width about summits for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    genome_fasta = args.genome_fasta
    peaks = load_labeled_peaks_from_beds(
        args.pos_regions, args.neg_regions, args.half_peak_width)
    model = KerasModel(peaks)
    results = ClassificationResults()
    for train, valid in peaks.iter_train_validation_subsets():
        fit_model = model.train_rSeqDNN_model(train, genome_fasta, './test')
        results.append(fit_model.evaluate_rSeqDNN_model(
        encode_peaks_sequence_into_binary_array(
                valid.peaks, genome_fasta), valid.labels))
    print 'Printing cross validation results:'
    print results

if __name__ == '__main__':
    main()
