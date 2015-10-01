import os
import gzip
import shutil

import multiprocessing

from scipy.stats.mstats import mquantiles

from pysam import FastaFile, TabixFile

from grit.lib.multiprocessing_utils import (
    fork_and_wait, ThreadSafeFile, Counter )

from peaks import (
    load_narrow_peaks, load_summit_centered_peaks, getFileHandle, 
    classify_chipseq_peak)

from motif_tools import (
    load_selex_models_from_db, 
    load_pwms_from_db, 
    score_region)

from DB import load_chipseq_peak_and_matching_DNASE_files_from_db

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
    
    def __init__(self, motifs, chipseq_peak_files):
        self.motifs = motifs
        self.chipseq_peak_files = chipseq_peak_files
        self.header_base = ['mean', 'max', 'q99', 'q95', 'q90', 'q75', 'q50']
        self.quantile_probs = [0.99, 0.95, 0.90, 0.75, 0.50]
        self.flank_sizes = [800, 500, 300]
        self.max_flank_size = max(self.flank_sizes)
        
        self.tabix_file_cache = {}
    
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

    def classify_chipseq_peak(self, sample, peak):
        return classify_chipseq_peak(
            self.chipseq_peak_files[sample], peak)

def extract_data_worker(ofp, peak_cntr, peaks, build_predictors, fasta):
    # reload the fasta file to make it thread safe
    fasta = FastaFile(fasta.filename)
    while True:
        index = peak_cntr.return_and_increment()
        if index >= len(peaks): break
        sample, peak = peaks[index]
        if peak.contig == 'chrM': continue
        try: 
            scores = build_predictors.build_summary_stats(peak, fasta)
            labels = build_predictors.classify_chipseq_peak(sample, peak)
        except Exception, inst: 
            print "ERROR", inst
            continue
        if index%50000 == 0:
            print "%i/%i" % (index, len(peaks))
        ofp.write("%s_%s\t%s\t%.4f\t%s\n" % (
            sample, "_".join(str(x) for x in peak).ljust(30), 
            ",".join(str(x) for x in labels), 
            peak[-1],
            "\t".join("%.4e" % x for x in scores)))
    return

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Score all in-vitro models for a particular TF.')

    parser.add_argument( '--selex-motif-id', help='Database SELEX motif ID')
    parser.add_argument( '--cisbp-motif-id', help='Database cisbp motif ID')

    parser.add_argument( '--ofprefix', type=str, default='peakscores',
                         help='Output file prefix (default peakscores)')

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
        assert False, "Must set either --slex-motif-id or --cisbp-motif-id"
    assert len(motifs) == 1
    print "Finished loading motifs."
    peak_fnames = load_chipseq_peak_and_matching_DNASE_files_from_db(
        motifs[0].tf_id)
    
    all_peaks = []
    chipseq_peak_filenames = {}
    for (sample_id, (chipseq_peaks_fnames, dnase_peaks_fnames)
            ) in peak_fnames.iteritems():
        chipseq_peak_filenames[sample_id] = chipseq_peaks_fnames
        for dnase_peaks_fname in dnase_peaks_fnames:
            with getFileHandle(dnase_peaks_fname) as fp:
                peaks = load_summit_centered_peaks(
                    load_narrow_peaks(fp, None),
                    400 )
            for peak in peaks:
                all_peaks.append((sample_id, peak))
    print "Finished loading peaks."

    ofname = "{prefix}.{motif_id}.txt.gz".format(
        prefix=args.ofprefix, 
        motif_id=motifs[0].motif_id
    )
    
    return (fasta, all_peaks, motifs, chipseq_peak_filenames, ofname)

def open_or_create_feature_file(
        fasta, all_peaks, motifs, chipseq_peak_filenames, ofname):
    try:
        return open(ofname)
    except IOError:
        print "Creating feature file '%s'" % ofname
        peak_cntr = Counter()
        build_predictors = BuildPredictorsFactory(
            motifs, chipseq_peak_filenames)
        with ThreadSafeFile(ofname + ".TMP", 'w') as ofp:
            ofp.write("\t".join(build_predictors.build_header()) + "\n")
            fork_and_wait(NTHREADS, extract_data_worker, (
                ofp, peak_cntr, all_peaks, build_predictors, fasta))
        
        input_fp = open(ofname + ".TMP")
        with gzip.open(ofname, 'wb') as ofp_compressed:
            shutil.copyfileobj(input_fp, ofp_compressed)
        input_fp.close()
        os.remove(ofname + '.TMP')
        return getFileHandle(ofname)

def main():
    fasta, all_peaks, motifs, chipseq_peak_filenames, ofname = parse_arguments()
    # check to see if this file is cached. If not, create it
    feature_fp = open_or_create_feature_file(
        fasta, all_peaks, motifs, chipseq_peak_filenames, ofname)
    print "Loading feature file '%s'" % ofname
    data = load_single_motif_data(feature_fp.name)
    res = estimate_cross_validated_error(data)
    print res
    return

if __name__ == '__main__':
    main()
