import os, sys
import numpy as np
from scipy.signal import convolve # fftconvolve as 

import cPickle as pickle

import psycopg2

from pysam import FastaFile, TabixFile

#import pyximport; pyximport.install()
#import score_seq

import multiprocessing
import grit
from grit.lib.multiprocessing_utils import fork_and_wait, ThreadSafeFile


from motif_tools import load_all_pwms_from_db as load_all_pwms, score_seq

def load_regions_in_bed(fp):
    regions = []
    for line in fp:
        data = line.split()
        contig = data[0]
        start = int(data[1])
        stop = int(data[2])
        regions.append((contig, start, stop))
    return regions

def score_regions_worker(ofp, genome, regions_queue, motifs):
    while regions_queue.qsize() > 0:
        region = regions_queue.get()
        print regions_queue.qsize()
        score_region(ofp, region, genome, motifs)

def score_regions(ofp, genome, regions, motifs):
    ofp.write("region".ljust(30) + " " + " ".join(
    (motif.tf_name + "_" + motif.tf_species[0]).ljust(15)
        for motif in motifs) + "\n")

    regions_queue = multiprocessing.Queue()
    regions_queue.cancel_join_thread()
    for region in regions:
        regions_queue.put(region)
    fork_and_wait(4, score_regions_worker, (ofp, genome, regions_queue, motifs))
    regions_queue.close()
    
def main():
    genome_fname = sys.argv[1]
    regions_fname = sys.argv[2]

    genome = FastaFile(genome_fname)
    print "Loaded genome"
    with open(regions_fname) as fp:
        regions = load_regions_in_bed(fp)
    print "Loaded regions"
    motifs = load_all_pwms()
    print "Loaded motifs"

    import cProfile
    
    with ThreadSafeFile('scores.txt', 'w') as ofp:
        score_regions(ofp, genome, regions, motifs)
    
main()
