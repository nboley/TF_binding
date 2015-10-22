import argparse

from pysam import FastaFile
from pyTFbindtools.peaks import getFileHandle

def evaluate_predictions(probs, y_validation):
    '''Evaluate the quality of deep bind predictions.
    '''
    preds = np.asarray(probs > 0.5, dtype='int')
    true_pos = (y_validation == 1)
    true_neg = (y_validation == -1)
    precision, recall, _ = precision_recall_curve(y_validation, probs)
    prc = np.array([recall,precision])
    auPRC = auc(recall, precision)
    auROC = roc_auc_score(y_validation, probs)
    classification_result = ClassificationResult(
        None, None, None, None, None, None,
        auROC, auPRC,
        np.sum(preds[true_pos] == 1), np.sum(true_pos),
        np.sum(preds[true_neg] == -1), np.sum(true_neg)
    )

    return classification_result

def get_data_for_deepsea_comparison(peaks_and_labels):
    '''exclude chr8-9 from training, put in validation
    '''
    samples = peaks_and_labels.samples
    validation_data = peaks_and_labels.subset_data(samples,
                                                   ['chr8', 'chr9'])
    training_data = peaks_and_labels.remove_data([],
                                                 ['chr8', 'chr9'])
    
    return [training_data, validation_data]

def init_prediction_script_argument_parser(description):
    parser = argparse.ArgumentParser(
        description=description)

    parser.add_argument('--tf-id',
                        help='TF to build model on')
    parser.add_argument('--annotation-id', type=int,
        help='genome annotation to get peak sequence from (default: hg19)')

    parser.add_argument('--pos-regions', type=getFileHandle,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle,
                        help='regions with negative labels')
    parser.add_argument('--genome-fasta', type=FastaFile,
                        help='genome file to get sequences')

    parser.add_argument('--half-peak-width', type=int, default=500,
                        help='half peak width about summits for training')
    parser.add_argument( '--skip-ambiguous-peaks', 
        default=False, action='store_true', 
        help='Skip regions that dont overlap the optimal peak set but do overlap a relaxed set')

    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to use')

    parser.add_argument( '--max-num-peaks-per-sample', type=int, 
        help='the maximum number of peaks to parse for each sample (used for debugging)')
    
    return parser
