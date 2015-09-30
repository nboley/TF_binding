import os
import argparse
import re
import gzip

import numpy as np
from scipy import misc
from collections import namedtuple
from pysam import FastaFile

from pyTFbindtools.peaks import load_summit_centered_peaks, load_narrow_peaks
from pyTFbindtools.sequence import code_seq

class PeaksAndLabels(list):
    pass

PeakAndLabel = namedtuple('PeakAndLabel', ['peak', 'sample', 'label'])

def getFileHandle(filename,mode="r"):
    if (re.search('.gz$',filename) or re.search('.gzip',filename)):
        if (mode=="r"):
            mode="rb";
        return gzip.open(filename,mode)
    else:
        return open(filename,mode)


def peaks_to_pngs(peaks, fasta, png_directory=None):
    '''Covert a list of peaks into pngs encoding their sequence. 
    
    input: peaks list, fasta file
    output: list of png filenames
    note: currently doesn't write reverse complements, that can be easily appended to this function
    '''
    if png_directory == None:
        png_directory = "./"

    png_fnames = {}
    for peak in peaks:
        seq = fasta.fetch(peak.contig, peak.start, peak.stop)
        seq_png_name = os.path.abspath(
            os.path.join(png_directory, peak.identifier + '.png'))
        misc.imsave(seq_png_name,
                    code_seq(seq),
                    format='png') 
    
        png_fnames[peak] = seq_png_name
    
    return png_fnames  

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

class PeakRegionData():
    def __init__(self, peaks_and_labels, fasta):
        assert isinstance(peaks_and_labels, PeaksAndLabels)
        self.coded_peaks_seqs = encode_peaks_sequence_into_binary_array(
            [pk for pk, sample, label in peaks_and_labels], fasta)
        self.peaks_and_labels = peaks_and_labels

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
    peaks_and_labels = PeaksAndLabels()
    for pos_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.pos_regions), args.half_peak_width):
        peaks_and_labels.append((pos_pk, 'sample', 1))
    for neg_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.neg_regions), args.half_peak_width):
        peaks_and_labels.append((neg_pk, 'sample', 0))
    peaks = PeakRegionData(peaks_and_labels, genome_fasta)
    print peaks.coded_peaks_seqs

if __name__ == '__main__':
    main()
