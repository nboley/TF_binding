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

def code_base(base):
    if base == 'A':
        return 0
    if base == 'a':
        return 0
    if base == 'C':
        return 1
    if base == 'c':
        return 1
    if base == 'G':
        return 2
    if base == 'g':
        return 2
    if base == 'T':
        return 3
    if base == 't':
        return 3
    return 4

def code_seq(seq):
    coded_seq = np.zeros((5,len(seq)))
    for i, base in enumerate(seq):
        coded_base = code_base(base)
        coded_seq[coded_base, i] = 1
    return coded_seq


from collections import namedtuple
PwmModel = namedtuple('PwmModel', [
    'tf_id', 'motif_id', 'tf_name', 'tf_species', 'pwm']) 

pickled_motifs_fname = os.path.join(
    os.path.dirname(__file__), 
    "../data/motifs/human_and_mouse_motifs.pickle.obj")

def load_regions_in_bed(fp):
    regions = []
    for line in fp:
        data = line.split()
        contig = data[0]
        start = int(data[1])
        stop = int(data[2])
        regions.append((contig, start, stop))
    return regions

def load_all_motifs():
    try: 
        with open(pickled_motifs_fname, "r") as fp:
            return pickle.load(fp)
    except: pass

    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
    SELECT tf_id, motif_id, tf_name, tf_species, pwm 
      FROM related_motifs_mv NATURAL JOIN pwms 
     WHERE tf_species in ('Mus_musculus', 'Homo_sapiens') 
       AND rank = 1;"""
    cur.execute(query)
    motifs = []
    for data in cur.fetchall():
        data = list(data)
        data[-1] = np.log2(np.array(data[-1]) + 1e-4)
        motifs.append( PwmModel(*data) )

    with open(pickled_motifs_fname, "w") as fp:
        return pickle.dump(motifs, fp)

    return motifs

def score_region(ofp, region, genome, motifs):
    seq = genome.fetch(*region)
    res = ["_".join(str(x) for x in region).ljust(30),]
    for motif in motifs:
        N_row = np.zeros((len(motif.pwm), 1)) + np.log2(0.25)
        extended_pwm = np.hstack((motif.pwm, N_row))
        coded_seq = code_seq(bytes(seq))
        scores = convolve(coded_seq, extended_pwm.T, mode='valid')
        res.append( ("%.4f" % (scores.sum()/len(scores))).ljust(15) )
    ofp.write(" ".join(res) + "\n")

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
    motifs = load_all_motifs()
    print "Loaded motifs"

    import cProfile
    
    with ThreadSafeFile('scores.txt', 'w') as ofp:
        score_regions(ofp, genome, regions, motifs)
    
main()
