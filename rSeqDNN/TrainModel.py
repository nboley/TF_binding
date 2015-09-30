import os
import argparse
import re
import gzip
from collections import namedtuple
from itertools import izip

import numpy as np
from scipy import misc

from pysam import FastaFile

from pyTFbindtools.peaks import load_summit_centered_peaks, load_narrow_peaks
from pyTFbindtools.sequence import code_seq
from pyTFbindtools.cross_validation import iter_train_validation_splits

PeakAndLabel = namedtuple('PeakAndLabel', ['peak', 'sample', 'label'])

def getFileHandle(filename,mode="r"):
    if (re.search('.gz$',filename) or re.search('.gzip',filename)):
        if (mode=="r"):
            mode="rb";
        return gzip.open(filename,mode)
    else:
        return open(filename,mode)

def encode_peaks_sequence_into_binary_array(peaks, fasta):
        # find the peak width
        pk_width = peaks[0].pk_width
        # make sure that the peaks are all the same width
        assert all(pk.pk_width == pk_width for pk in peaks)
        data = 0.25 * np.ones((len(peaks), 4, pk_width))
        for i, pk in enumerate(peaks):
            seq = fasta.fetch(pk.contig, pk.start, pk.stop)
            coded_seq = code_seq(seq)
            data[i] = coded_seq[0:4,:]
        return data

class PeaksAndLabels():
    def __iter__(self):
        return (
            PeakAndLabel(pk, sample, label) 
            for pk, sample, label 
            in izip(self.peaks, self.samples, self.labels)
        )
    
    def __init__(self, peaks_and_labels):
        # split the peaks and labels into separate columns. Also
        # keep track of the distinct samples and contigs
        self.peaks = []
        self.samples = []
        self.labels = []
        self.sample_ids = set()
        self.contigs = set()
        for pk, sample, label in peaks_and_labels:
            self.peaks.append(pk)
            self.samples.append(sample)
            self.labels.append(label)
            self.sample_ids.add(sample)
            self.contigs.add(pk.contig)
        
        # turn the list of labels into a numpy array
        self.labels = np.array(self.labels, dtype=int)
    
    # One hot encode the sequence in each peak
    def build_coded_seqs(self, fasta):
        return encode_peaks_sequence_into_binary_array(
            self.peaks, fasta)
    
    def subset_data(self, sample_names, contigs):
        '''return data covering sample+names and contigs
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.sample in sample_names
                and pk_and_label.peak.contig in contigs
            )

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
    peaks_and_labels = []
    for pos_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.pos_regions), args.half_peak_width):
        peaks_and_labels.append((pos_pk, 'sample', 1))
    for neg_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.neg_regions), args.half_peak_width):
        peaks_and_labels.append((neg_pk, 'sample', 0))
    peaks = PeaksAndLabels(peaks_and_labels)
    train = peaks.subset_data(['sample'], ['chr1'])
    print train.build_coded_seqs(genome_fasta)

if __name__ == '__main__':
    main()
