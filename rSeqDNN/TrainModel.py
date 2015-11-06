import os
import numpy as np
from collections import OrderedDict
from pysam import FastaFile

try:
    from rSeqDNN import init_prediction_script_argument_parser
except:
    from __init__ import init_prediction_script_argument_parser

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds,
    load_labeled_peaks_from_fastas,
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import ClassificationResults

from KerasModel import KerasModel, encode_peaks_sequence_into_binary_array

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for estimating rSeqDNN accuracy via cross validation')
    parser.add_argument('--only-test-one-fold', 
                        default=False, action='store_true',
        help='Only evaluate on a single cross validation fold.')
    parser.add_argument('--random-seed', type=int, default=1701,
                    help='random seed. 1701 by default.')
    parser.add_argument('--single-celltype', default=False, action='store_true',
                    help='train and validate on single celltypes')
    parser.add_argument('--validation-contigs', type=str, default=None,
                    help='to validate on chr1 and chr4, input chr1,chr4')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    train_parser = subparsers.add_parser('train', help='training help')
    test_parser = subparsers.add_parser('test', help='testing help')
    train_parser.add_argument('--model-prefix', default='trainedmodel',
        help='Trained models will be written to (model_prefix).foldnum.h5"')
    train_parser.add_argument('--use-model-file',
                        default=False, action='store_true',
        help='pickle model during training to avoid recompiling.')
    train_parser.add_argument('--model-definition-file', type=str, default=None,
                    help='JSON file containing model architecture.')
    test_parser.add_argument('--model-file', default=None, type=str,
        help='pickled model file, defaults to default KerasModel')
    test_parser.add_argument('--weights-file', type=str, required=True,
        help='model weights file')
    
    args = parser.parse_args()
    if args.validation_contigs is not None:
        args.validation_contigs = set(args.validation_contigs.split(','))
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
        assert args.pos_sequences is None and args.neg_sequences is None, \
            "It doesnt make sense to set both --tf-id and either --pos-sequences or --neg-sequences"
        assert args.annotation_id is not None, \
            "--annotation-id must be set if --tf-id is set"
        assert args.genome_fasta is None, \
            "if --tf-id is set the genome fasta must be specified by the --annotation-id"

        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            args.tf_id,
            args.annotation_id,
            args.half_peak_width, 
            args.max_num_peaks_per_sample, 
            include_ambiguous_peaks=True)
    else:
        if args.pos_regions != None:
            assert args.neg_regions != None, \
            "--neg-regions must be set"
            peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
        elif args.pos_sequences != None:
            assert args.neg_sequences != None, \
            "--neg-sequences must be set"
            peaks_and_labels = load_labeled_peaks_from_fastas(
                args.pos_sequences, args.neg_sequences, args.half_peak_width)
        else:
            raise ValueError('either --tf-id or (--pos-regions and --neg-regions)\
            or (--pos-sequences and --neg-sequences) must be set')

    if args.command=='train':
        return ( peaks_and_labels, 
                 genome_fasta, 
                 args.model_prefix, 
                 args.only_test_one_fold,
                 args.use_model_file,
                 args.model_definition_file,
                 args.random_seed,
                 args.single_celltype,
                 args.validation_contigs), args.command
    elif args.command=='test':
        return ( peaks_and_labels,
                 genome_fasta,
                 args.only_test_one_fold,
                 args.model_file,
                 args.weights_file,
                 args.random_seed,
                 args.single_celltype,
                 args.validation_contigs), args.command

import cPickle as pickle

def main():
    args, command = parse_args()
    if command=='train':
        ( peaks_and_labels, 
          genome_fasta, 
          model_ofname_prefix, 
          only_test_one_fold,
          use_model_file,
          model_definition_file,
          random_seed,
          single_celltype,
          validation_contigs
      ) = args
    elif command=='test':
        ( peaks_and_labels,
          genome_fasta,
          only_test_one_fold,
          model_fname,
          weights_fname,
          random_seed,
          single_celltype,
          validation_contigs
      ) = args
    np.random.seed(random_seed) # fix random seed
    results = ClassificationResults()
    clean_results = ClassificationResults()
    training_data = OrderedDict()
    validation_data = OrderedDict()
    if command=='train':
        if (model_definition_file is not None):
            model = KerasModel(peaks_and_labels,
                               model_def_file=model_definition_file)
        else:
            model = KerasModel(peaks_and_labels)
    elif command=='test':
        fit_model = KerasModel(peaks_and_labels, model_fname=model_fname)
        if model_fname is None:
            fit_model.compile()
        fit_model.model.load_weights(weights_fname)
    for fold_index, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets(validation_contigs,
                                                               single_celltype)):
        if command=='train':
            fit_model = model.train(
                train, 
                genome_fasta, 
                '%s.%i.hd5' % (model_ofname_prefix, fold_index+1),
                use_model_file,
                train.can_use_seq)
        
        clean_res = fit_model.evaluate_peaks_and_labels(
            valid, 
            genome_fasta,
            valid.can_use_seq,
            filter_ambiguous_labels=True)
        print "CLEAN:", clean_res
        clean_results.append(clean_res)

        if not command=='test': ## TODO: load/provide threshold
            res = fit_model.evaluate_peaks_and_labels(
                valid, 
                genome_fasta,
                valid.can_use_seq,
                filter_ambiguous_labels=False,
                plot_fname=("ambig.fold%i.png" % fold_index))
            print "FULL:", res
            results.append(res)
        
        if not command=='test':
            training_data[fold_index] = [train.sample_ids, train.contigs]
        validation_data[fold_index] = [valid.sample_ids, valid.contigs]
        
        if only_test_one_fold: break

    print 'Printing validation results for each fold:'
    for fold_index in validation_data.keys():
        if not command=='test':
            print 'training data: ', training_data[fold_index]
        print 'validation data: ', validation_data[fold_index]
        print 'CLEAN:', clean_results[fold_index]
        if not command=='test':
            print 'FULL:', results[fold_index]
    print 'Printing validation results over all folds:'
    print 'CLEAN:'
    print clean_results
    if not command=='test':
        print 'FULL:'
        print results
        

if __name__ == '__main__':
    main()

