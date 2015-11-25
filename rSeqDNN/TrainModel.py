import os
import numpy as np
from collections import OrderedDict
from pysam import FastaFile

try:
    from rSeqDNN import init_prediction_script_argument_parser
except:
    from __init__ import init_prediction_script_argument_parser

from pyTFbindtools.peaks import (
    FastaPeaksAndLabels,
    load_labeled_peaks_from_beds,
    load_labeled_peaks_from_fastas,
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import ClassificationResults

from KerasModel import (
    KerasModel,
    encode_peaks_sequence_into_binary_array,
    load_model
)

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
    train_parser.add_argument('--use-cached-model',
                        default=False, action='store_true',
        help='pickle model during training to avoid recompiling.')
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
                args.pos_sequences, args.neg_sequences, args.max_num_peaks_per_sample)
        else:
            raise ValueError('either --tf-id or (--pos-regions and --neg-regions)\
            or (--pos-sequences and --neg-sequences) must be set')

    main_args = ( peaks_and_labels,
                  genome_fasta,
                  args.only_test_one_fold,
                  args.random_seed,
                  args.single_celltype,
                  args.validation_contigs)
    if args.command=='train':
        command_args = ( args.model_prefix,
                         args.use_cached_model)
    elif args.command=='test':
        command_args = ( args.model_file,
                         args.weights_file)
    return  main_args, command_args, args.command

def main_train(main_args, train_args):
    ( peaks_and_labels,
      genome_fasta,
      only_test_one_fold,
      random_seed,
      single_celltype,
      validation_contigs ) = main_args
    model_ofname_prefix, use_cached_model = train_args
    np.random.seed(random_seed) # fix random seed
    results = ClassificationResults()
    clean_results = ClassificationResults()
    training_data = OrderedDict()
    validation_data = OrderedDict()
    model = KerasModel(peaks_and_labels,
                       use_cached_model=use_cached_model)
    if isinstance(peaks_and_labels, FastaPeaksAndLabels):
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets())
    else:
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets(
            validation_contigs, single_celltype))
    for fold_index, (train, valid) in enumerate(train_validation_subsets):
        fit_model = model.train(
            train,
            genome_fasta,
            '%s.%i.hd5' % (model_ofname_prefix, fold_index+1),
            use_cached_model)
        clean_res = fit_model.evaluate_peaks_and_labels(
            valid,
            genome_fasta,
            filter_ambiguous_labels=True)
        if not isinstance(peaks_and_labels, FastaPeaksAndLabels):
            clean_res.train_samples = train.sample_ids
            clean_res.train_chromosomes = train.contigs
            clean_res.validation_samples = valid.sample_ids
            clean_res.validation_chromosomes = valid.contigs
        clean_results.append(clean_res)
        if not isinstance(peaks_and_labels, FastaPeaksAndLabels):
            res = fit_model.evaluate_peaks_and_labels(
                valid,
                genome_fasta,
                filter_ambiguous_labels=False,
                plot_fname=("ambig.fold%i.png" % fold_index))
            res.train_samples = train.sample_ids
            res.train_chromosomes = train.contigs
            res.validation_samples = valid.sample_ids
            res.validation_chromosomes = valid.contigs
            results.append(res)
        if only_test_one_fold: break
    print 'CLEAN VALIDATION RESULTS'
    for res in clean_results:
        print res
    print clean_results
    print 'FULL VALIDATION RESULTS'
    for res in results:
        print res
    print results

def main_test(main_args, test_args):
    ( peaks_and_labels,
      genome_fasta,
      only_test_one_fold,
      random_seed,
      single_celltype,
      validation_contigs ) = main_args
    model_fname, weights_fname = test_args
    np.random.seed(random_seed) # fix random seed
    clean_results = ClassificationResults()
    validation_data = OrderedDict()
    fit_model = KerasModel(peaks_and_labels)
    if model_fname is not None:
        fit_model.model = load_model(model_fname)
    else:
        fit_model.compile()
    fit_model.model.load_weights(weights_fname)
    if isinstance(peaks_and_labels, FastaPeaksAndLabels):
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets())
    else:
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets(
            validation_contigs, single_celltype))
    for fold_index, (train, valid) in enumerate(train_validation_subsets):
        clean_res = fit_model.evaluate_peaks_and_labels(
            valid,
            genome_fasta,
            filter_ambiguous_labels=True)
        if not isinstance(peaks_and_labels, FastaPeaksAndLabels):
            clean_res.validation_samples = valid.sample_ids
            clean_res.validation_chromosomes = valid.contigs
        clean_results.append(clean_res)
        if only_test_one_fold: break
    print 'CLEAN VALIDATION RESULTS'
    for res in clean_results:
        print res
    print clean_results

if __name__ == '__main__':
    main_args, command_args, command  = parse_args()
    if command=='train':
        main_train(main_args, command_args)
    elif command=='test':
        main_test(main_args, command_args)
