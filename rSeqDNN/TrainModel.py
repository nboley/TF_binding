import os

from pysam import FastaFile

from rSeqDNN import init_prediction_script_argument_parser

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB
)

from pyTFbindtools.cross_validation import ClassificationResults

from KerasModel import KerasModel, encode_peaks_sequence_into_binary_array

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for estimating rSeqDNN accuracy via cross validation')
    parser.add_argument('--model-prefix', default='trainedmodel',
        help='Trained models will be written to (model_prefix).foldnum.h5"')
    parser.add_argument('--only-test-one-fold', 
                        default=False, action='store_true',
        help='Only evaluate on a single cross validation fold.')

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
            args.skip_ambiguous_peaks,
            include_ambiguous_peaks=True)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
    
    return ( peaks_and_labels, 
             genome_fasta, 
             args.model_prefix, 
             args.only_test_one_fold )

import cPickle as pickle

def main():
    ( peaks_and_labels, genome_fasta, model_ofname_prefix, only_test_one_fold
    ) = parse_args()
    model = KerasModel(peaks_and_labels)
    results = ClassificationResults()
    for fold_index, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets()):
        fit_model = model.train_rSeqDNN_model(
            train, 
            genome_fasta, 
            '%s.%i.hd5' % (model_ofname_prefix, fold_index+1))
        results.append(fit_model.evaluate(valid, genome_fasta, True))
        print results[-1]
        if only_test_one_fold: break
    
    print 'Printing cross validation results:'
    for res in results:
        print res
    print results

if __name__ == '__main__':
    main()
