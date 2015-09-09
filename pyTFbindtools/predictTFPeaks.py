
from pysam import FastaFile
from peaks import load_narrow_peaks

from motif_tools import load_pwms_from_db, score_region

NTHREADS = 1

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate tf binding from sequence and chromatin accessibility data.')

    parser.add_argument( '--fasta', type=file, required=True,
        help='Fasta containing genome sequence.')

    parser.add_argument( '--tf-names', nargs='+',
                         help='A list of human TF names')

    parser.add_argument( '--peaks', type=file,  required=True,
        help='Narrowpeak file containing peaks to predict on.')

    parser.add_argument( '--threads', '-t', default=1, type=int,
                         help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads

    global PLOT
    PLOT = args.plot

    fasta = FastaFile(args.fasta.name)

    peaks = load_narrow_peaks(args.peaks.name)
    
    # load the motifs
    motifs = load_pwms_from_db(args.td_names)
    
    # load the chipseq peaks
    
    return (motifs, fasta, peaks)

def main():
    motifs, fasta, peaks = parse_arguments()
    pass
