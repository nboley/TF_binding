import random
random.seed(1701)
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
    merge_peaks_and_labels,
    load_and_label_peaks_from_beds,
    load_labeled_peaks_from_beds,
    load_labeled_peaks_from_fastas,
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    load_conservation_fnames_from_DB,
    load_matching_dnase_cut_fnames_from_DB,
    load_matching_dnase_foldchange_fnames_from_DB
)
from pyTFbindtools.cross_validation import (
    ClassificationResult, ClassificationResults
)
from get_signal import (
    get_peaks_signal_arrays, get_peaks_signal_arrays_by_samples,
    merge_sample_specific_bigwigs
)
from KerasModel import KerasModel, load_model, KerasModelMultitask
from grid_search import MOESearch

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
    parser.add_argument('--include-model-report', default=False, action='store_true',
                    help='plot model predictions')
    parser.add_argument('--accessibility-track',  default=None, type=str,
                        dest='bigwig_features',
                        help='bigwig file to get features')
    parser.add_argument('--exclude-sequence',  action='store_true',
                        help='ignore sequence features')
    parser.add_argument('--include-dnase-foldchange',  action='store_true',
                        help='integrate dnase features when loading from db')
    parser.add_argument('--conservation-features',  default=None, type=str,
                        help='bigwig file to get conservation features')
    parser.add_argument('--include-conservation',  action='store_true',
                        help='integrate conservation features when loading from db')
    parser.add_argument('--dnase-cut-features',  default=None, type=str,
                        help='bigwig file to get features')
    parser.add_argument('--include-dnase-cuts',  action='store_true',
                        help='integrate dnase cut features when loading from db')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    train_parser = subparsers.add_parser('train', help='training help')
    test_parser = subparsers.add_parser('test', help='testing help')
    train_parser.add_argument('--model-prefix', default='trainedmodel',
        help='Trained models will be written to (model_prefix).foldnum.h5"')
    train_parser.add_argument('--model-file', default=None, type=str,
        help='pickled model architecture to train')
    train_parser.add_argument('--weights-file', type=str, default=None,
        help='model weights to finetune')
    train_parser.add_argument('--multi-mode', action='store_true',
                    help='treat different features as separate modes')
    train_parser.add_argument('--jitter-peaks-by', type=str, default=None,
                              help='10,-15,5 will jitter peaks by 10, -15, and 5 ')
    train_parser.add_argument('--target-metric', type=str, default='recall_at_05_fdr',
                              help='metric used for model selection. '\
                              'supported options: auROC, auPRC, F1, recall_at_05_fdr'\
                              'recall_at_10_fdr, recall_at_05_fdr.')
    train_parser.add_argument('--run-grid-search', default=False, action='store_true',
                    help='performs large scale hyper parameter selection')
    test_parser.add_argument('--model-file', type=str, required=True,
        help='pickled model file, defaults to default KerasModel')
    test_parser.add_argument('--weights-file', type=str, required=True,
        help='model weights file')

    args = parser.parse_args()
    if args.validation_contigs is not None:
        args.validation_contigs = set(args.validation_contigs.split(','))
    if args.command == 'train':
        if args.jitter_peaks_by is not None:
            try:
                args.jitter_peaks_by = map(int, args.jitter_peaks_by.split(','))
            except Exception as e:
                raise ValueError('invalid jitter values, jitters must be integers!')
                raise e
        assert (args.target_metric in ClassificationResult._fields), \
            "Invalid target metric, see supported metrics."
    assert args.annotation_id is not None or args.genome_fasta is not None, \
        "Must set either --annotation-id or --genome-fasta"
    if args.genome_fasta is None:
        assert args.annotation_id is not None
        from pyTFbindtools.DB import load_genome_metadata
        genome_fasta = FastaFile(
            load_genome_metadata(args.annotation_id).filename) 
    else:
        genome_fasta = args.genome_fasta
    if args.exclude_sequence: genome_fasta=None

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
        # general preprocessing
        peaks_and_labels = peaks_and_labels.remove_ambiguous_labeled_entries()
        peaks_and_labels = peaks_and_labels.filter_by_contig_edge(9000, genome_fasta) # removes peaks in contigs edges
        if args.include_dnase_foldchange:
            args.bigwig_features = load_matching_dnase_foldchange_fnames_from_DB(args.tf_id, args.annotation_id)
        if args.include_conservation:
            args.conservation_features = load_conservation_fnames_from_DB()
        if args.include_dnase_cuts:
            args.dnase_cut_features = load_matching_dnase_cut_fnames_from_DB(args.tf_id, args.annotation_id)
    else:
        if args.pos_regions != None:
            if args.neg_regions != None:
            #assert args.neg_regions != None, \
            #"--neg-regions must be set"
                peaks_and_labels = load_labeled_peaks_from_beds(
                    args.pos_regions, args.neg_regions,
                    args.half_peak_width, args.max_num_peaks_per_sample)
            elif args.background_regions:
                peaks_and_labels = load_and_label_peaks_from_beds(
                    args.background_regions, args.pos_regions, args.ambiguous_regions,
                    args.bin_size, args.flank_size, args.max_num_peaks_per_sample)
            else:
                raise RuntimeError("--neg-regions or --background-regions must be set when --pos-regions is set")
            if len(np.shape(peaks_and_labels.labels))==1:
                peaks_and_labels = peaks_and_labels.remove_ambiguous_labeled_entries()
            peaks_and_labels = peaks_and_labels.filter_by_contig_edge(9000, genome_fasta)
        elif args.pos_sequences != None:
            assert args.neg_sequences != None, \
            "--neg-sequences must be set"
            peaks_and_labels = load_labeled_peaks_from_fastas(
                args.pos_sequences, args.neg_sequences, args.max_num_peaks_per_sample)
        else:
            raise ValueError('either --tf-id or (--pos-regions and --neg-regions)\
            or (--pos-sequences and --neg-sequences) must be set')

    #sample_specific_bigwigs = merge_bigwigs_dictionaries(args.bigwig_features, args.dnase_cut_features)
    main_args = ( peaks_and_labels,
                  genome_fasta,
                  args.only_test_one_fold,
                  args.random_seed,
                  args.single_celltype,
                  args.validation_contigs,
                  args.include_model_report,
                  args.conservation_features,
                  args.dnase_cut_features,
                  args.bigwig_features)
    if args.command=='train':
        command_args = ( args.model_prefix,
                         args.model_file,
                         args.weights_file,
                         args.multi_mode,
                         args.jitter_peaks_by,
                         args.target_metric,
                         args.run_grid_search)
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
      validation_contigs,
      include_model_report,
      conservation_features,
      dnase_cut_features,
      bigwig_features ) = main_args
    ( model_ofname_prefix,
      model_fname,
      weights_fname,
      multi_mode,
      jitter_peaks_by,
      target_metric,
      run_grid_search ) = train_args
    bigwig_features = dnase_cut_features
    np.random.seed(random_seed) # fix random seed
    results = ClassificationResults()
    clean_results = ClassificationResults()
    training_data = OrderedDict()
    validation_data = OrderedDict()
    if isinstance(peaks_and_labels, FastaPeaksAndLabels):
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets())
    else:
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets(
            validation_contigs=validation_contigs, single_celltype=single_celltype))
    for fold_index, (train, valid) in enumerate(train_validation_subsets):
        print ('Testing Samples: %s' % valid.sample_ids)
        # preprocess peaks, get signal, initialize model, train, test
        fitting_data, stopping_data = next(train.iter_train_validation_subsets())
        print('Fitting Samples: %s' % fitting_data.sample_ids)
        print('Stopping Samples: %s' % stopping_data.sample_ids)
        # preprocess peaks
        if jitter_peaks_by is not None:
            fitting_data_positives = fitting_data.filter_by_label(1)
            jittered_positives = merge_peaks_and_labels(
                fitting_data_positives.jitter_peaks(jitter)
                for jitter in jitter_peaks_by)
            fitting_data = merge_peaks_and_labels((fitting_data, jittered_positives))
        # get signal arrays
        if isinstance(bigwig_features, dict):
            (fitting_signal_arrays,
             fitting_labels,
             fitting_scores ) = get_peaks_signal_arrays_by_samples(fitting_data, genome_fasta, bigwig_features,
                                                                   sample_independent_bigwigs=conservation_features,
                                                                   reverse_complement=True,
                                                                   return_labels=True, return_scores=True)
            (stopping_signal_arrays,
             stopping_labels,
             stopping_scores ) = get_peaks_signal_arrays_by_samples(stopping_data, genome_fasta, bigwig_features,
                                                                    sample_independent_bigwigs=conservation_features,
                                                                    reverse_complement=False,
                                                                    return_labels=True, return_scores=True)
        else:
            if isinstance(bigwig_features, dict):
                bigwig_features = bigwig_features[train.sample_ids[0]]
            fitting_signal_arrays = get_peaks_signal_arrays(
                fitting_data.peaks, genome_fasta, bigwig_features, reverse_complement=True)
            fitting_labels = np.concatenate((fitting_data.labels, fitting_data.labels))
            fitting_scores = np.concatenate((fitting_data.scores, fitting_data.scores))
            stopping_signal_arrays = get_peaks_signal_arrays(
                stopping_data.peaks, genome_fasta, bigwig_features, reverse_complement=False)
            stopping_labels = stopping_data.labels
            stopping_scores = stopping_data.scores
        signal_arrays_shapes = [np.shape(signal_array) for signal_array in fitting_signal_arrays]
        print 'signal_arrays_shapes: ', signal_arrays_shapes
        print "shape of the labels: ", np.shape(fitting_labels)
        if run_grid_search:
            param_grid  = dict(zip(['num_conv_layers','num_conv', 'conv_width', 'maxpool_size', 'l1_decay', 'dropout'],
                                   [[1,5], [10, 60], [8, 30], [5,50], [0, 0.01], [0, 0.5]]))
            param_types = dict(zip(['num_conv_layers','num_conv', 'conv_width', 'maxpool_size', 'l1_decay', 'dropout'],
                                   ['disc', 'disc', 'disc', 'disc', 'cont', 'cont']))
            fixed_param = dict(zip(['peaks_and_labels', 'target_metric'],[peaks_and_labels, target_metric]))
            conditional_param = dict(zip(['maxpool_stride'],['maxpool_size']))
            model_search = MOESearch(KerasModel, param_grid, param_types,
                                           fixed_param=fixed_param, conditional_param=conditional_param)
            fit_method = 'train'
            fit_param = dict(zip(['data', 'genome_fasta', 'ofname', 'jitter_peaks_by'],
                                 [train, genome_fasta, '%s.%i' % (model_ofname_prefix, fold_index+1), jitter_peaks_by]))
            score_method = 'score'
            score_param = {}
            fit_model_search = model_search.fit(fit_method, fit_param, score_method, score_param, max_iter=60,
                                                      minimize=False, email_updates_to='johnnyisraeli@gmail.com')
            fit_model = fit_model_search.best_estimator_
            print 'hyper parameters selected by grid search:'
            print fit_model_search.best_grid_param_
            print 'resulting in selection score: %f' % fit_model_search.best_score_
        else:
            if model_fname is not None and weights_fname is not None:
                model = KerasModel(model_fname=model_fname)
                model.model.load_weights(weights_fname)
            else:
                if len(np.shape(fitting_labels))==2 and np.shape(fitting_labels)[-1]>1:
                    print "instantiating KerasModelMultitask..."
                    model = KerasModelMultitask(np.shape(fitting_labels)[-1], signal_arrays_shapes, target_metric=target_metric, multi_mode=multi_mode)
                else:
                    model = KerasModel(signal_arrays_shapes, target_metric=target_metric, multi_mode=multi_mode)
            model_ofname_infix = valid.samples[0] if len(peaks_and_labels.samples[0])>0 else str(fold_index+1)
            fit_model = model.train(
                fitting_signal_arrays, fitting_labels, fitting_scores,
                stopping_signal_arrays, stopping_labels, stopping_scores,
                '%s.%s' % (model_ofname_prefix, model_ofname_infix))
        if isinstance(bigwig_features, dict):
            (valid_signal_arrays,
             valid_labels,
             valid_scores ) = get_peaks_signal_arrays_by_samples(valid, genome_fasta, bigwig_features,
                                                                 sample_independent_bigwigs=conservation_features,
                                                                 reverse_complement=False,
                                                                 return_labels=True, return_scores=True)
        else:
            valid_signal_arrays = get_peaks_signal_arrays(
                valid.peaks, genome_fasta, bigwig_features, reverse_complement=False)
            valid_labels = valid.labels
            valid_scores = valid.scores
        clean_res = fit_model.evaluate_peaks_and_labels(
            valid_signal_arrays,
            valid_labels,
            include_ambiguous_labels=False)
        if not isinstance(peaks_and_labels, FastaPeaksAndLabels):
            if isinstance(clean_res, list):
                for res in clean_res:
                    res.train_samples = train.sample_ids
                    res.train_chromosomes = train.contigs
                    res.validation_samples = valid.sample_ids
                    res.validation_chromosomes = valid.contigs
            else:
                clean_res.train_samples = train.sample_ids
                clean_res.train_chromosomes = train.contigs
                clean_res.validation_samples = valid.sample_ids
                clean_res.validation_chromosomes = valid.contigs
        clean_results.append(clean_res)
        if (valid_labels == -1).any() and isinstance(fit_model, KerasModel):
            res = fit_model.evaluate_peaks_and_labels(
                valid_signal_arrays,
                valid_labels,
                include_ambiguous_labels=True,
                scores=valid_scores,
                plot_fname=("ambig.fold%i.png" % fold_index))
            res.train_samples = train.sample_ids
            res.train_chromosomes = train.contigs
            res.validation_samples = valid.sample_ids
            res.validation_chromosomes = valid.contigs
            results.append(res)
            if include_model_report:
                fit_model.classification_report(
                    stopping_signal_arrays,
                    stopping_labels,
                    "%s.fold%i" % (
                        model_ofname_prefix, fold_index),
                    stopping_scores)
        if only_test_one_fold: break
    print 'CLEAN VALIDATION RESULTS'
    for clean_result in clean_results:
        if isinstance(clean_result, list):
            for res in clean_res:
                print res
        else:
            print res
    #print clean_results
    if len(results) > 0:
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
      validation_contigs,
      include_model_report,
      conservation_features,
      dnase_cut_features,
      bigwig_features ) = main_args
    model_fname, weights_fname = test_args
    np.random.seed(random_seed) # fix random seed
    clean_results = ClassificationResults()
    validation_data = OrderedDict()
    fit_model = KerasModel(model_fname=model_fname)
    fit_model.model.load_weights(weights_fname)
    if isinstance(peaks_and_labels, FastaPeaksAndLabels):
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets())
    else:
        train_validation_subsets = list(peaks_and_labels.iter_train_validation_subsets(
            validation_contigs=validation_contigs, single_celltype=single_celltype))
    for fold_index, (train, valid) in enumerate(train_validation_subsets):
        if len(peaks_and_labels.sample_ids) > 1:
            (valid_signal_arrays,
             valid_labels,
             valid_scores ) = get_peaks_signal_arrays_by_samples(valid, genome_fasta, bigwig_features,
                                                                 sample_independent_bigwigs=conservation_features,
                                                                 reverse_complement=False,
                                                                 return_labels=True, return_scores=True)
        else:
            if isinstance(bigwig_features, dict):
                bigwig_features = bigwig_features[train.sample_ids[0]]
            valid_signal_arrays = get_peaks_signal_arrays(
                valid.peaks, genome_fasta, bigwig_features, reverse_complement=False)
            valid_labels = valid.labels
            valid_scores = valid.scores
        clean_res = fit_model.evaluate_peaks_and_labels(
            valid_signal_arrays,
            valid_labels,
            include_ambiguous_labels=False)
        if not isinstance(peaks_and_labels, FastaPeaksAndLabels):
            if isinstance(clean_res, list):
                for res in clean_res:
                    res.train_samples = train.sample_ids
                    res.train_chromosomes = train.contigs
                    res.validation_samples = valid.sample_ids
                    res.validation_chromosomes = valid.contigs
            else:
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
