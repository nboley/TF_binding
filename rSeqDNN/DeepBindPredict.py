import os
import argparse
import numpy as np

from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import ClassificationResults


def encode_peaks_sequence_into_fasta_file(validation_data, fasta, fasta_filename, labels_filename):
    '''writes data peaks sequence into file and saves labels
    '''
    np.savetxt(labels_filename, validation_data.labels)
    peaks = validation_data.peaks
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    with open(fasta_file, 'w') as wf:
        for i, pk in enumerate(peaks):
            seq = fasta.fetch(pk.contig, pk.start, pk.stop)
            wf.write('>'+str(i)+'\n')
            wf.write(seq+'\n')

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for testing rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')

    parser.add_argument('--tf-id',
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
    
    return peaks_and_labels, args.genome_fasta, args.output_fasta_filename_prefix


def main():
    peaks_and_labels, genome_fasta, fasta_filename_prefix = parse_args()
    results = ClassificationResults()
    for train, valid in peaks_and_labels.iter_train_validation_subsets():
        print 'writing sequences and labels to files..'
        for sample in valid.sample_ids:
            for contig in valid.contigs:
                subset_fasta_filename = '_'.join([fasta_filename_prefix, sample, contig]) + '.fa'
                print subset_fasta_file
                subset_labels_filename = '.'.join([subset_fasta_filename, 'labels'])
                subset_validation_data = valid.subset_data([sample], [contig])
                encode_peaks_sequence_into_fasta_file(subset_valid,
                                                      genome_fasta,
                                                      subset_fasta_filename,
                                                      subset_labels_filename)
        print 'done!'
        break


if __name__ == '__main__':
    main()
