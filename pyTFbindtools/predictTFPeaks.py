import os, sys

from collections import defaultdict

from pysam import FastaFile, TabixFile
from peaks import load_narrow_peaks

from motif_tools import load_pwms_from_db, load_selex_models_from_db, score_region

import multiprocessing
import grit
from grit.lib.multiprocessing_utils import fork_and_wait, ThreadSafeFile

from scipy.stats.mstats import mquantiles

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

def load_summary_stats(peak, fasta, motifs):
    header_base = ['mean', 'max', 'q99', 'q95', 'q90', 'q75', 'q50']
    header = ['region',] + ["label_%s" % motif.tf_name for motif in motifs] + ["access_score",]
    quantile_probs = [0.99, 0.95, 0.90, 0.75, 0.50]
    summary_stats = []
    seq_peak = (peak.contig, 
                peak.start+peak.summit-800, 
                peak.start+peak.summit+800)
    region_motifs_scores = score_region(seq_peak, fasta, motifs)
    for motif, motif_scores in zip(motifs, region_motifs_scores):
        header.extend("%s_%i_%s" % (motif.tf_name, 1600, label) 
                      for label in header_base)
        summary_stats.append(motif_scores.mean()/len(motif_scores))
        summary_stats.append(motif_scores.max())
        for quantile in mquantiles(motif_scores, prob=quantile_probs):
            summary_stats.append(quantile)
        motif_scores = motif_scores[500:-500]
        header.extend("%s_%i_%s" % (motif.tf_name, 600, label) 
                      for label in header_base)
        summary_stats.append(motif_scores.mean()/len(motif_scores))
        summary_stats.append(motif_scores.max())
        for quantile in mquantiles(motif_scores, quantile_probs):
            summary_stats.append(quantile)
    return header, summary_stats

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
        sample, peak = peaks[index]
        if peak.contig == 'chrM': continue
        try: 
            header, scores = load_summary_stats(peak, fasta, motifs)
            labels = classify_peak(peak, sample, motifs)
        except: 
            continue
        if index%1000 == 0:
            print "%i/%i" % (index, len(peaks))
        ofp.write("%s_%s\t%s\t%.4f\t%s\n" % (
            sample, "_".join(str(x) for x in peak).ljust(30), 
            "\t".join(str(x) for x in labels), 
            peak[-1],
            "\t".join("%.4e" % x for x in scores)))
    return

def parse_arguments():
    def find_dnase_peak_fnames(args):
        assert args.peaks == None or args.roadmap_sample_ids == None, \
            "if --roadmap-sample-ids is set --peaks is inferred from it (so it doesn't make sense to set both)" 
        peak_fnames = []
        if args.peaks != None:
            assert args.roadmap_sample_ids == None
            for fp in args.peaks:
                fname = fp.name
                sample_id = os.path.basename(fname).split('-')[0]
                peak_fnames.append((sample_id, fname))
        else:
            for sample_id in args.roadmap_sample_ids:
                # check if the sample id is valid
                try:
                    assert ( sample_id[0] == 'E' and int(sample_id[1:]) >= 0 
                             and int(sample_id[1:]) < 1000 )
                except:
                    raise ValueError, "Invalid Roadmap Sample ID '%s' (must be in form E000)" % sample_id 
                fname = "/mnt/data/epigenomeRoadmap/peaks/consolidated/narrowPeak/%s-DNase.macs2.narrowPeak.gz" % sample_id
                try: 
                    fp = open(fname)
                    fp.close()
                except IOError:
                    raise ValueError, "Can't open DNASE peak file for roadmap sample id '%s'" % sample_id
                peak_fnames.append((sample_id, fname))
        return peak_fnames

    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate tf binding from sequence and chromatin accessibility data.')

    parser.add_argument( '--fasta', type=file, required=True,
        help='Fasta containing genome sequence.')

    parser.add_argument( '--tf-names', nargs='+',
                         help='A list of human TF names')

    parser.add_argument( '--peaks', type=file, nargs='*',
        help='Narrowpeak file(s) containing peaks to predict on.')
    parser.add_argument( '--roadmap-sample-ids', nargs='*',
        help='Specify the peak fieles with roadmap sample ids (eg E123)')

    score_type_options = ['ddG','logOdds']
    parser.add_argument( '--score-type', type=str, default='ddG',
                         choices=score_type_options,
                         help='Motif scoring method.')

    parser.add_argument( '--ofprefix', type=str, default='peakscores',
                         help='Output file prefix (default peakscores)')

    parser.add_argument( '--threads', '-t', default=1, type=int,
                         help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads

    fasta = FastaFile(args.fasta.name)
    print "Finished initializing fasta."
    
    # load the motifs
    if args.score_type == 'ddG':
        motifs = load_selex_models_from_db(args.tf_names)
    elif args.score_type == 'logOdds':
        motifs = load_pwms_from_db(args.tf_names)
    print "Finished loading motifs."
    
    # load all of the peaks
    peak_samples_and_fnames = find_dnase_peak_fnames(args)
    all_peaks = []
    for sample_id, peaks_fname in peak_samples_and_fnames:
        peaks = load_narrow_peaks(peaks_fname, None) #None) #1000
        for peak in peaks:
            all_peaks.append((sample_id, peak))
    print "Finished loading peaks."

    ofname = "{prefix}.{scoring_type}.{motifs}.{sample_ids}.txt".format(
        prefix=args.ofprefix, scoring_type=args.score_type,
        motifs="_".join(sorted(motif.tf_name for motif in motifs)),
        sample_ids="_".join(sorted(
            sample for sample, fname in peak_samples_and_fnames))
    )
    
    return (motifs, fasta, all_peaks, ofname)

def main():
    motifs, fasta, peaks, output_fname = parse_arguments()
    print "Writing output to '%s'" % output_fname
    peak_cntr = Counter()
    header, stats = load_summary_stats(peaks[100][1], fasta, motifs)
    with ThreadSafeFile(output_fname, 'w') as ofp:
        ofp.write("\t".join(header) + "\n")
        fork_and_wait(NTHREADS, extract_data_worker, (
            ofp, peak_cntr, motifs, fasta, peaks))

main()
