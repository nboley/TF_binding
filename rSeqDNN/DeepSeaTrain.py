import os
import argparse
import numpy as np
import subprocess
import tempfile
import time  
import multiprocessing
from math import log
from scipy import io

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

from rSeqDNN import evaluate_predictions, init_prediction_script_argument_parser
from KerasModel import encode_peaks_sequence_into_binary_array


def encode_peaks_and_labels_in_matfile(fname, peaks_and_labels, fasta):
    '''write dictionary with feature and label array
    encode features with encode_peaks_sequence_into_binary_array
    '''
    # encode peaks sequence into binary array
    features_and_labels = {}
    features = encode_peaks_sequence_into_binary_array(peaks_and_labels.peaks,
                                                       fasta)
    num_examples = len(features)
    labels = np.reshape(peaks_and_labels.labels, (num_examples, 1)) 
    # get key name prefix
    # TODO: make this user-specified after deepsea patch
    data_subset = fname.split('.')[0]
    # save dictionary with features and labels in matfile
    features_and_labels[data_subset+'xdata'] = features
    features_and_labels[data_subset+'data'] = labels
    io.savemat(fname, features_and_labels)

def download_and_fix_deepsea_training():
    '''patch deepsea training software
    '''
    pass

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for testing rSeqDNN')

    parser.add_argument('--output-directory', required=True,
                        help='output directory for deepsea results')
    
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
            args.skip_ambiguous_peaks)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
        
    return ( peaks_and_labels, 
             genome_fasta,
             args.tf_id, 
             args.output_directory, 
             args.threads)

def train_deepsea(input_list):
    '''
    runs all functions
    '''
    ( training_data,
      validation_data, 
      sample, 
      contig, 
      genome_fasta_fname, 
      tf_id, 
      output_directory ) = input_list
    genome_fasta = FastaFile(genome_fasta_fname)
    
    initial_wd = os.getcwd()
    ( training_fitting,
      training_stopping ) = next(training_data.iter_train_validation_subsets())
    # save training data in mat files for deepsea
    encode_peaks_and_labels_in_matfile('train.mat',
                                       training_fitting,
                                       genome_fasta)
    encode_peaks_and_labels_in_matfile('valid.mat',
                                       training_stopping,
                                       genome_fasta)

    # define deepsea training settings
    learning_rate = '1'
    learning_rate_decay = '8e-7'
    weight_decay= '1e-6'
    momentum = '0.9'
    stdv = '0.05'
    set_device = '1' # gpu device
    # TODO: user specified
    window_size = '1000'
    max_kernel_norm = '0.9'
    batch_size = '16'
    l1_sparsity = '1e-8'
    training_size = str(len(training_fitting.labels))
    validation_size = str(len(training_stopping.labels))
    # train deepsea
    command = ' '.join(['th main.lua',
                        '-save', output_directory,
                        '-LearningRate', learning_rate,
                        '-LearningRateDecay', learning_rate_decay,
                        '-weightDecay', weight_decay,
                        '-momentum', momentum,
                        '-stdv', stdv,
                        '-setDevice', set_device,
                        '-windowsize', window_size,
                        '-max_kernel_norm', max_kernel_norm,
                        '-batchSize', batch_size,
                        '-L1Sparsity', l1_sparsity,
                        '-training_size', training_size,
                        '-validation_size', validation_size])
    process = subprocess.Popen(command, shell=True)

def main():
    download_and_fix_deepsea_training()
    ( peaks_and_labels, 
      genome_fasta, 
      tf_id, 
      output_directory, 
      num_threads ) = parse_args()
    inputs = []
    for count, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets()):        
        for sample in valid.sample_ids:
            for contig in valid.contigs:
                inputs.append([train,
                               valid, 
                               sample, 
                               contig, 
                               genome_fasta.filename,
                               tf_id, 
                               output_directory])
        break

    for args in inputs:
        train_deepsea(args)
        break

if __name__ == '__main__':
    main()
