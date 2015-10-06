import os
import gzip
import shutil

import multiprocessing

from scipy.stats.mstats import mquantiles

from pysam import FastaFile, TabixFile

from grit.lib.multiprocessing_utils import (
    fork_and_wait, ThreadSafeFile, Counter )

from peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    getFileHandle, 
    classify_chipseq_peak)

from motif_tools import (
    load_selex_models_from_db, 
    load_pwms_from_db, 
    score_region)

from analyze_data import load_single_motif_data, estimate_cross_validated_error

NTHREADS = 1
    
class BuildPredictorsFactory(object):
    def build_header(self):
        header = ['region',] + [
            "label__%s" % motif.motif_id for motif in self.motifs] + [
                "access_score",]
        for motif in self.motifs:
            for flank_size in self.flank_sizes:
                header.extend("%s__%i__%s" % (
                    motif.motif_id, 2*flank_size, label) 
                              for label in self.header_base)
        return header
    
    def __init__(self, motifs):
        self.motifs = motifs
        self.header_base = ['mean', 'max', 'q99', 'q95', 'q90', 'q75', 'q50']
        self.quantile_probs = [0.99, 0.95, 0.90, 0.75, 0.50]
        self.flank_sizes = [800, 500, 300]
        self.max_flank_size = max(self.flank_sizes)
        
    def build_summary_stats(self, peak, fasta):
        seq_peak = (peak.contig, 
                    peak.start+peak.summit-self.max_flank_size, 
                    peak.start+peak.summit+self.max_flank_size)
        region_motifs_scores = score_region(seq_peak, fasta, self.motifs)
        summary_stats = []
        for motif, motif_scores in zip(self.motifs, region_motifs_scores):
            for flank_size in self.flank_sizes:
                inner_motif_scores = motif_scores[
                    self.max_flank_size-flank_size
                    :(self.max_flank_size-flank_size)+2*flank_size+1]
                summary_stats.append(
                    inner_motif_scores.mean()/len(inner_motif_scores))
                summary_stats.append(inner_motif_scores.max())
                for quantile in mquantiles(
                        inner_motif_scores, prob=self.quantile_probs):
                    summary_stats.append(quantile)
        return summary_stats

def extract_data_worker(ofp, peak_cntr, peaks, build_predictors, fasta):
    # reload the fasta file to make it thread safe
    fasta = FastaFile(fasta.filename)
    while True:
        index = peak_cntr.return_and_increment()
        if index >= len(peaks): break
        labeled_peak = peaks[index]
        try: 
            scores = build_predictors.build_summary_stats(
                labeled_peak.peak, fasta)
        except Exception, inst: 
            print "ERROR", inst
            continue
        if index%50000 == 0:
            print "%i/%i" % (index, len(peaks))
        ofp.write("%s_%s\t%s\t%.4f\t%s\n" % (
            labeled_peak.sample, 
            "_".join(str(x) for x in labeled_peak.peak).ljust(30), 
            labeled_peak.label, 
            labeled_peak.peak[-1],
            "\t".join("%.4e" % x for x in scores)))
    return

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Score all in-vitro models for a particular TF.')

    parser.add_argument( '--selex-motif-id', 
        help='Database SELEX motif ID')
    parser.add_argument( '--cisbp-motif-id', 
        help='Database cisbp motif ID')

    parser.add_argument( '--half-peak-width', type=int, default=400,
        help='Example accessible region peaks to be +/- --half-peak-width bases from the summit (default: 400)')
    parser.add_argument( '--ofprefix', type=str, default='peakscores',
        help='Output file prefix (default peakscores)')

    parser.add_argument( '--max-num-peaks-per-sample', type=int, 
        help='the maximum number of peaks to parse for each sample (used for debugging)')

    parser.add_argument( '--threads', '-t', default=1, type=int, 
        help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads

    fasta_fname = "/mnt/data/annotations/by_release/hg19.GRCh37/hg19.genome.fa"
    fasta = FastaFile(fasta_fname)
    print "Finished initializing fasta."

    if args.selex_motif_id != None:
        motifs = load_selex_models_from_db(motif_ids=[args.selex_motif_id,])
    elif args.cisbp_motif_id != None:
        motifs = load_pwms_from_db(motif_ids=[args.cisbp_motif_id,])
    else:
        assert False, "Must set either --selex-motif-id or --cisbp-motif-id"
    assert len(motifs) == 1
    motif = motifs[0]
    print "Finished loading motifs."

    ofname = "{prefix}.{motif_id}.txt.gz".format(
        prefix=args.ofprefix, 
        motif_id=motifs[0].motif_id
    )
    
    return (fasta, motif, ofname, 
            args.half_peak_width, args.max_num_peaks_per_sample)

def open_or_create_feature_file(
        fasta, motif, ofname, 
        half_peak_width,
        max_n_peaks_per_sample=None):
    try:
        return open(ofname)
    except IOError:
        print "Creating feature file '%s'" % ofname
        labeled_peaks = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            motif.tf_id, 
            half_peak_width=half_peak_width, 
            max_n_peaks_per_sample=max_n_peaks_per_sample)
        print "Finished loading peaks."

        peak_cntr = Counter()
        build_predictors = BuildPredictorsFactory([motif,])
        with ThreadSafeFile(ofname + ".TMP", 'w') as ofp:
            ofp.write("\t".join(build_predictors.build_header()) + "\n")
            fork_and_wait(NTHREADS, extract_data_worker, (
                ofp, peak_cntr, labeled_peaks, build_predictors, fasta))
        
        input_fp = open(ofname + ".TMP")
        with gzip.open(ofname, 'wb') as ofp_compressed:
            shutil.copyfileobj(input_fp, ofp_compressed)
        input_fp.close()
        os.remove(ofname + '.TMP')
        return getFileHandle(ofname)

def main():
    (fasta, motif, ofname, half_peak_width, max_num_peaks_per_sample
        ) = parse_arguments()
    # check to see if this file is cached. If not, create it
    feature_fp = open_or_create_feature_file(
        fasta, motif, ofname, 
        half_peak_width=half_peak_width,
        max_n_peaks_per_sample=max_num_peaks_per_sample)
    print "Loading feature file '%s'" % ofname
    data = load_single_motif_data(feature_fp.name)
    res = estimate_cross_validated_error(data)
    with open(ofname + ".summary", "w") as ofp:
        print >> ofp, res.all_data
    return

if __name__ == '__main__':
    main()
