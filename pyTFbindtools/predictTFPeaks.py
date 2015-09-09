import os, sys

from collections import defaultdict

from pysam import FastaFile, TabixFile
from peaks import load_narrow_peaks

from motif_tools import load_pwms_from_db, score_region

import multiprocessing
import grit
from grit.lib.multiprocessing_utils import fork_and_wait, ThreadSafeFile

NTHREADS = 1

term_name_RMID_mapping = {
    'A549': 'E114',
    'T-cell-acute-lymphoblastic-leukemia': 'E115',
    'GM12878': 'E116',
    'HeLa-S3': 'E117',
    'HepG2': 'E118',
    'mammary-epithelial-cell': 'E119',
    'skeletal-muscle-myoblast': 'E120',
    'myotube': 'E121',
    'endothelial-cell-of-umbilical-vein': 'E122',
    'K562': 'E123' 
}

class Counter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

    def return_and_increment(self):
        with self.lock:
            rv = self.val.value
            self.val.value += 1
        return rv

RMID_term_name_mapping = {}
for term_name, RMID in term_name_RMID_mapping.items():
    RMID_term_name_mapping[RMID] = term_name

def load_tf_peaks_fnames(
        base_dir='/mnt/data/TF_binding/in_vivo/nathans_ENCODE_tfs/'):
    tf_peaks = defaultdict(list)
    fnames = os.listdir(base_dir)
    for fname in fnames:
        if not fname.endswith(".bgz"): continue
        meta_data = fname.split(".")[0].split("_")
        factor_name = meta_data[0]
        peak_type = meta_data[2]
        term_name = meta_data[3]
        if peak_type != 'UniformlyProcessedPeakCalls': continue
        tf_peaks[(factor_name, term_name)].append(
            os.path.join(base_dir, fname))
    return tf_peaks

tf_peak_fnames = load_tf_peaks_fnames()

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

    fasta = FastaFile(args.fasta.name)

    peaks = load_narrow_peaks(args.peaks.name, None) #100)
    
    # load the motifs
    motifs = load_pwms_from_db(args.tf_names)

    # load the chipseq peaks
    
    return (motifs, fasta, peaks)

def load_summary_scores(peak, fasta, motifs):
    seq_peak = (peak.contig, 
                peak.start+peak.summit-800, 
                peak.start+peak.summit+800)
    region_scores = score_region(seq_peak, fasta, motifs)
    return [x.mean()/len(x) for x in region_scores]

def classify_peak(peak, sample, motifs):
    pc_peak = (peak.contig, 
               peak.start+peak.summit-300, 
               peak.start+peak.summit+300)
    nc_peak = (peak.contig, 
               peak.start+peak.summit-2000, 
               peak.start+peak.summit+2000)
    status = []
    for motif in motifs:
        fname = tf_peak_fnames[
            (motif.tf_name, RMID_term_name_mapping[sample])][0]
        fp = TabixFile(fname)
        if peak[0] not in fp.contigs: 
            status.append(0)
            continue
        pc_peaks = list(fp.fetch(*pc_peak))
        if len(pc_peaks) > 0:
            status.append(1)
            continue
        nc_peaks = list(fp.fetch(*nc_peak))
        if len(nc_peaks) == 0:
            status.append(-1)
        else:
            status.append(0)
    return status

def extract_data_worker(ofp, peak_cntr, motifs, fasta, peaks):
    # reload the fasta file to make it thread safe
    fasta = FastaFile(fasta.filename)
    while True:
        index = peak_cntr.return_and_increment()
        if index >= len(peaks): break
        peak = peaks[index]
        if peak.contig == 'chrM': continue
        score = load_summary_scores(peak, fasta, motifs)[0]
        label = classify_peak(peak, 'E122', motifs)[0]
        print "%i/%i" % (index, len(peaks)), score, label
        ofp.write("%s\t%s\t%s\n" % (
            "_".join(str(x) for x in peak).ljust(30), score, label))
    return

def main():
    motifs, fasta, peaks = parse_arguments()
    peak_cntr = Counter()
    ofp = ThreadSafeFile('output.txt', 'w')
    fork_and_wait(24, extract_data_worker, (ofp, peak_cntr, motifs, fasta, peaks))

main()
