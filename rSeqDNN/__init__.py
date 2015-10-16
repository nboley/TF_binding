import argparse

from pysam import FastaFile
from pyTFbindtools.peaks import getFileHandle

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
