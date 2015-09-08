import os, sys
import numpy as np

import cPickle as pickle

import psycopg2

from pysam import FastaFile, TabixFile

import pyximport; pyximport.install()
import score_seq

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

    for region in regions:
        seq = genome.fetch(*region)
        print region
        for motif in motifs:
            score = score_seq.score_seq_from_pwm(bytes(seq), motif.pwm)
            print region, score
            return
    
main()
