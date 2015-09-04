import os, sys
sys.path.insert(0, "/users/nboley/src/TF_binding/")
from motif_tools import Motif, logistic, R, T
from selex import code_sequence

import numpy as np

from scipy.signal import fftconvolve

import psycopg2
import psycopg2.extras

from pysam import FastaFile, TabixFile

def load_all_motifs():
    conn = psycopg2.connect("host=mitra dbname=cisbp")
    cur = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
    #query = "select * from related_motifs_mv NATURAL JOIN pwms where tf_species = 'Mus_musculus' and rank = 1;"
    query = "select * from related_motifs_mv NATURAL JOIN pwms where tf_species = 'Homo_sapiens' and rank = 1;"
    cur.execute(query)
    motifs = []
    for res in cur.fetchall():
        motif = Motif(res.tf_id, res.tf_name, res.pwm)
        motifs.append(motif)
    return motifs

def load_tfname_tfid_mapping():
    conn = psycopg2.connect("host=mitra dbname=cisbp")
    cur = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
    query = "select array_agg(tf_id), tf_name from tfs where tf_species = 'Homo_sapiens' group by tf_name order by tf_name;"
    cur.execute(query)
    tfname_id_map = {}
    for res in cur.fetchall():
        tfname_id_map[res.tf_name] = res.array_agg
    return tfname_id_map



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

        for i, base in enumerate(seq.upper()):
            coded_seq[base][len(seq)-i-1] = 1
        RC_scores = np.zeros(len(seq)-len(score_mat)+1, dtype='float32')
        for base, base_scores in zip('TCGA', score_mat.T):
            scores += np.convolve(coded_seq[base], base_scores, mode='valid')

        max_scores = np.vstack((scores, RC_scores)).max(0)
        motif_scores.append( max_scores.mean() )

    return np.array(motif_scores)

ChIPseq_peaks = TabixFile(
    "/mnt/data/TF_binding/in_vivo/ENCODE/CHiP_seq_peaks/human_ENCODE_TFS.bed.gz")

def load_overlapping_peaks(chrm, start, stop):
    overlapping_tfs = set()
    try: 
        for res_str in ChIPseq_peaks.fetch(chrm, start, stop):
            overlapping_tfs.add(res_str.split()[-1])
    except:
        pass
    return overlapping_tfs

def main():
    genome_fname = sys.argv[1]
    regions_fname = sys.argv[2]
    
    genome = FastaFile(genome_fname)
    print "Loaded genome"
    motifs = load_all_motifs()
    tfname_id_map = load_tfname_tfid_mapping()
    print "Loaded Motifs"
    with open(regions_fname) as fp:
        regions = load_regions_in_bed(fp)
    print "Loaded regions"

    with open(os.path.basename(regions_fname)+".peaks.txt", "w") as ofp:
        ofp.write("\t".join(["region".ljust(30),] + [
            motif.name for motif in motifs]) +"\n")
        ofp.write("\t".join(["region".ljust(30),] + [
            motif.factor for motif in motifs]) +"\n")
        for i, region in enumerate(regions):
            print i, region
            overlapping_peaks = load_overlapping_peaks(*region)
            motif_overlap_scores = []
            for motif in motifs:
                if motif.factor in overlapping_peaks:
                    motif_overlap_scores.append(1)
                else:
                    motif_overlap_scores.append(0)
            ofp.write("%s\t%s\n" % (
                      "_".join(map(str, region)).ljust(30), 
                      "\t".join("%i" % motif_overlap 
                                for motif_overlap in motif_overlap_scores)))
    print "Finished building peak overlap matrix"

    with open(os.path.basename(regions_fname) + ".TFscores.txt", "w") as ofp:
        ofp.write("\t".join(["region".ljust(30),] + [
            motif.name for motif in motifs]) +"\n")
        ofp.write("\t".join(["region".ljust(30),] + [
            motif.factor for motif in motifs]) +"\n")
        for i, region in enumerate(regions):
            if i%100 == 0: print i, len(regions), os.path.basename(regions_fname)
            seq = genome.fetch(*region).upper()
            try: scores = score_region(motifs, seq)
            except: continue
            ofp.write("%s\t%s\n" % (
                      "_".join(map(str, region)).ljust(30), 
                      "\t".join("%.4f" % score for score in scores)))
    print "Finished building score matrix"


main()
