import os, sys
from collections import defaultdict
import numpy
import math

from pysam import FastaFile

import multiprocessing
import Queue

import gzip

T = 300
R = 1.987-3 # in kCal
R = 8.314e-3

REG_LEN = 10000

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RC_base_map = {'A': 3, 'C': 2, 'G': 1, 'T': 0}

class ProcessSafeOPStream( object ):
    def __init__( self, writeable_obj ):
        self.writeable_obj = writeable_obj
        self.lock = multiprocessing.Lock()
        self.name = self.writeable_obj.name
        return
    
    def write( self, data ):
        self.lock.acquire()
        self.writeable_obj.write( data )
        self.writeable_obj.flush()
        self.lock.release()
        return
    
    def close( self ):
        self.writeable_obj.close()

ofp = ProcessSafeOPStream(open("output.bedgraph", "w"))
of_summits = ProcessSafeOPStream(open("output.summits", "w"))

def logit(x):
    return math.log(x) - math.log(1-x)

def logistic(x):
    e_x = math.exp(x)
    return e_x/(1+e_x)

def load_motif(motif_str):
    # load the motif data
    lines = motif_str.strip().split("\n")
    factor = lines[0].split("_")[0]
    motif_data = numpy.zeros((len(lines)-1, 4), dtype=float)
    consensus_energy = 0.0
    for i, line in enumerate(lines[1:]):
        row = numpy.array([logit(1e-2/2 + (1-1e-2)*float(x)) 
                           for x in line.split()[1:]])
        max_val = row.max()
        consensus_energy += max_val
        row -= max_val
        motif_data[i, :] = row

    stringency = 1e-3
    max_energy = logit(1-stringency)

    mean_energy = sum(row.sum()/4 for row in motif_data) 
    
    scale_factor = max_energy/consensus_energy

    consensus_energy, motif_data = (
        consensus_energy*scale_factor, motif_data*scale_factor )
    
    return factor, consensus_energy, motif_data

def load_all_motifs(fp):
    fp.seek(0)
    raw_motifs = fp.read().split(">")
    motifs = defaultdict(list)
    for motif_str in raw_motifs:
        if len(motif_str) == 0: continue
        factor, consensus_energy, motif_data = load_motif(motif_str)
        motifs[factor].append( [consensus_energy, motif_data] )
    return motifs

def load_and_process_motifs():
    with open(sys.argv[1]) as fp:
        factors_and_motifs = load_all_motifs(fp)
    for factor, motifs in factors_and_motifs.items():
        for cons_energy, motif in motifs:
            mean_energy = sum(row.sum()/4 for row in motif) 
            mean_occ = logistic(mean_energy/(R*T))
            cons_occ = logistic(cons_energy/(R*T))
            ratio = cons_occ/mean_occ
            print "%e" % ratio, factor, cons_energy, cons_occ, mean_energy, mean_occ, ratio

def score_seq(seq, motif):
    seq = seq.upper()
    cons_energy, motif_data = motif
    motif_len = len(motif_data)
    for offset in xrange(len(seq) - motif_len):
        subseq = seq[offset:offset+motif_len]
        assert motif_len == len(subseq)
        score = cons_energy
        RC_score = cons_energy
        if 'N' in subseq: continue
        for i, base in enumerate(subseq):
            score += motif_data[i][base_map[base]]
            RC_score += motif_data[motif_len-i-1][RC_base_map[base]]
        yield offset, max(score, RC_score)

def build_wig_worker(fasta_fname, regions, motif):
    fasta = FastaFile(fasta_fname)
    n_regions = regions.qsize()
    output = []
    output_summits = []
    while not regions.empty():
        try: chrm, start, stop, summit = regions.get(timeout=0.1)
        except Queue.Empty: return
        print >> sys.stderr, "Processing %s:%i-%i\t(%i/%i)" % (
            chrm, start, stop, regions.qsize(), n_regions)
        seq = fasta.fetch(chrm, start, stop)
        max_score = -1e9
        best_pos = -1
        lines = []
        for pos, score in score_seq(seq, motif):
            score = logistic(score)
            if score > max_score:
                max_score = score
                best_pos = pos
            output.append( "%s\t%i\t%i\t%.2f\n" % (
                chrm, start + pos, start + pos + 1, score) )
        output_summits.append("%i\t%i\t%.2f\t%.2f\n" % (
            best_pos, summit, 
            best_pos/float(stop-start), summit/float(stop-start)))

    ofp.write("".join(output))
    of_summits.write("".join(output))
    return

def put_all_regions_into_queue(fasta, motif_data):
    proc_queue = multiprocessing.Queue() #JoinableQueue()
    for ref, ref_len in zip(fasta.references, fasta.lengths):
        if ref != 'chr20': continue
        for reg_start in numpy.arange(0, ref_len, REG_LEN, dtype=int):
            reg_stop = reg_start+REG_LEN+len(motif_data)
            proc_queue.put((ref, reg_start, reg_stop))

def put_peaks_into_queue(fname):
    proc_queue = multiprocessing.Queue() #JoinableQueue()
    with gzip.open(fname) as fp:
        for i, line in enumerate(fp):
            if line.startswith("track"): continue
            if i > 1000: break
            data = line.split()
            chrm = data[0]
            start = int(data[1])
            stop = int(data[2])
            summit = int(data[9])
            proc_queue.put((chrm, start, stop, summit))
    return proc_queue

def main():
    with open(sys.argv[3]) as fp:
        factor, consensus_energy, motif_data = load_motif(fp.read().strip())
    print factor
    print consensus_energy
    print logistic(consensus_energy)

    mean_energy = sum(row.sum()/4 for row in motif_data) 
    print mean_energy
    print logistic(mean_energy)

    print motif_data
    #return
    fasta_fname = sys.argv[1]
    
    proc_queue = put_peaks_into_queue(sys.argv[2])

    pids = []
    for i in xrange(24):
        pid = os.fork()
        if pid == 0:
            build_wig_worker(fasta_fname, proc_queue, (consensus_energy, motif_data))
            os._exit(0)
        else:
            pids.append(pid)
    for pid in pids:
        os.wait(pid, 0)
main()
