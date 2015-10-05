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
    
    return peaks_and_labels, args.genome_fasta, args.model_prefix

def main():
    peaks_and_labels, genome_fasta, model_ofname_prefix = parse_args()
    model = KerasModel(peaks_and_labels)
    results = ClassificationResults()
    for fold_index, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets()):
        fit_model = model.train_rSeqDNN_model(
            train, 
            genome_fasta, 
            '%s.%i.hd5' % (model_ofname_prefix, fold_index+1))
        results.append(fit_model.evaluate_rSeqDNN_model(
            encode_peaks_sequence_into_binary_array(
                valid.peaks, genome_fasta), valid.labels))
        break
    print 'Printing cross validation results:'
    print results

if __name__ == '__main__':
    main()
