import os, sys
sys.path.insert(0, "/users/nboley/src/TF_binding/")
from motif_tools import Motif, logistic, R, T
from selex import code_sequence

import numpy as np

from scipy.signal import fftconvolve

import psycopg2
import psycopg2.extras

from pysam import FastaFile

def load_all_motifs():
    conn = psycopg2.connect("host=mitra dbname=cisbp")
    cur = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
    query = "select * from related_motifs_mv NATURAL JOIN pwms where tf_species = 'Mus_musculus' and rank = 1;"
    cur.execute(query)
    motifs = []
    for res in cur.fetchall():
        motif = Motif(res.tf_id, res.tf_name, res.pwm)
        motifs.append(motif)
    return motifs

def load_regions_in_bed(fp):
    regions = []
    for line in fp:
        data = line.split()
        contig = data[0]
        start = int(data[1])
        stop = int(data[2])
        regions.append((contig, start, stop))
    return regions

def score_region(motifs, seq):
    # code the sequence
    coded_seq = {}
    for base in 'ACGT':
        coded_seq[base] = np.zeros(len(seq), dtype='float32')
    for i, base in enumerate(seq.upper()):
        coded_seq[base][i] = 1
    coded_RC_seq = {}
    for base in 'TGCA':
        coded_seq[base] = np.zeros(len(seq), dtype='float32')


    motif_scores = []
    for motif in motifs:
        score_mat = motif.motif_data
        scores = np.zeros(len(seq)-len(score_mat)+1, dtype='float32')
        for base, base_scores in zip('ACGT', score_mat.T):
            scores += np.convolve(coded_seq[base], base_scores, mode='valid')
        occs = logistic(-16 + (motifs[0].consensus_energy + scores)/(R*T))

        for i, base in enumerate(seq.upper()):
            coded_seq[base][len(seq)-i-1] = 1
        scores = np.zeros(len(seq)-len(score_mat)+1, dtype='float32')
        for base, base_scores in zip('TCGA', score_mat.T):
            scores += np.convolve(coded_seq[base], base_scores, mode='valid')
        RC_occs = logistic(-16 + (motifs[0].consensus_energy + scores)/(R*T))

        max_occs = np.vstack((occs, RC_occs)).max(0)
        motif_scores.append( max_occs.mean() )

    return np.array(motif_scores)

def main():
    genome_fname = sys.argv[1]
    regions_fname = sys.argv[2]

    genome = FastaFile(genome_fname)
    print "Loaded genome"
    motifs = load_all_motifs()
    print "Loaded Motifs"
    with open(regions_fname) as fp:
        regions = load_regions_in_bed(fp)
    print "Loaded regions"
    with open(os.path.basename(regions_fname) + ".TFscores.txt", "w") as ofp:
        ofp.write("\t".join(["region",] + [motif.name for motif in motifs]) +"\n")
        ofp.write("\t".join(["region",] + [motif.factor for motif in motifs]) +"\n")
        for i, region in enumerate(regions):
            if i%1000 == 0: print i, len(regions), os.path.basename(regions_fname)
            seq = genome.fetch(*region).upper()
            try: scores = score_region(motifs, seq)
            except: continue
            ofp.write("%s\t%s\n" % (
                      "_".join(map(str, region)), 
                      "\t".join("%.4f" % score for score in scores)))

main()
