import argparse

from pysam import FastaFile
from pyTFbindtools.peaks import getFileHandle

import theano

import os

def config_theano(device, num_omp_threads=1):
    assert device[:3] in ('cpu', 'gpu')
    os.environ['THEANO_FLAGS'] = "'device=%s'" % device
    os.environ['OMP_NUM_THREADS'] = str(num_omp_threads)
    #theano.config.device = device
    theano.config.optimizer = 'fast_compile'

    # if the device is a cpu, assume we're in debuggin mode and set flags
    # appropriately
    if device == 'cpu':
        theano.config.mode='FAST_COMPILE'
        theano.config.exception_verbosity='high'
        theano.config.openmp=True        
    else:
        theano.config.mode='FAST_RUN'
        #theano.config.cuda.root='/usr/local/cuda'
        theano.config.floatX='float32'
        theano.config.warn_float64='warn'
        theano.config.assert_no_cpu_op='warn'
    return

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
    parser.add_argument( '--include-ambiguous-peaks', 
        default=False, action='store_true', 
        help='Include regions that dont overlap the optimal peak set but do overlap a relaxed set')

    parser.add_argument('--model-definition-file', type=str, default=None,
        		help='JSON file containing model architecture.')

    #parser.add_argument('--device', type=str, required=True,
    #                    choices=['cpu', 'gpu', 'gpu1', 'gpu2', 'gpu3', 'gpu4'],
    #                    help='Use the specified device.')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads to use')

    parser.add_argument( '--max-num-peaks-per-sample', type=int, 
        help='the maximum number of peaks to parse for each sample (used for debugging)')
    
    # set up the theano environemnt as soon as possible
    #args, _ = parser.parse_known_args()
    #config_theano(args.device, args.threads)
    
    return parser
