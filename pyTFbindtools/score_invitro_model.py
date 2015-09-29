import os
import multiprocessing

from scipy.stats.mstats import mquantiles

from pysam import FastaFile, TabixFile

from grit.lib.multiprocessing_utils import fork_and_wait, ThreadSafeFile

from peaks import load_narrow_peaks

from motif_tools import (
    load_selex_models_from_db, 
    load_pwms_from_db, 
    load_chipseq_peak_and_matching_DNASE_files_from_db,
    score_region)

NTHREADS = 1

class Counter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.Value('i', initval)
        self.lock = multiprocessing.Lock()

    def return_and_increment(self):
        with self.lock:
            rv = self.val.value
            self.val.value += 1
        return rv

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

    def classify_peak(self, sample, peak):
        pid = os.getppid()
        pc_peak = (peak.contig, 
                   peak.start+peak.summit-300, 
                   peak.start+peak.summit+300)
        nc_peak = (peak.contig, 
                   peak.start+peak.summit-2000, 
                   peak.start+peak.summit+2000)
        status = []
        for motif in self.motifs:
            motif_status = []
            for fname in self.chipseq_peak_files[sample]:
                try: 
                    fp = self.tabix_file_cache[(pid, fname)]
                except KeyError:
                    fp = TabixFile(fname)
                    self.tabix_file_cache[(pid, fname)] = fp

                if peak[0] not in fp.contigs: 
                    motif_status.append(0)
                    continue
                pc_peaks = list(fp.fetch(*pc_peak))
                if len(pc_peaks) > 0:
                    motif_status.append(1)
                    continue
                nc_peaks = list(fp.fetch(*nc_peak))
                if len(nc_peaks) == 0:
                    motif_status.append(-1)
                else:
                    motif_status.append(0)
            status.append(",".join(str(x) for x in motif_status))
        return status

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
            labels = build_predictors.classify_peak(sample, peak)
        except Exception, inst: 
            print "ERROR", inst
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

    motifs = load_selex_models_from_db(motif_ids=[args.selex_motif_id,])
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
            peaks = load_narrow_peaks(dnase_peaks_fname, None) #1000
            for peak in peaks:
                all_peaks.append((sample_id, peak))
    print "Finished loading peaks."

    ofname = "{prefix}.{motif_id}.txt".format(
        prefix=args.ofprefix, 
        motif_id=args.selex_motif_id
    )
    
    return (fasta, all_peaks, motifs, chipseq_peak_filenames, ofname)

def main():
    fasta, all_peaks, motifs, chipseq_peak_filenames, ofname = parse_arguments()
    print "Writing output to '%s'" % ofname
    peak_cntr = Counter()
    build_predictors = BuildPredictorsFactory(motifs, chipseq_peak_filenames)
    with ThreadSafeFile(ofname, 'w') as ofp:
        ofp.write("\t".join(build_predictors.build_header()) + "\n")
        fork_and_wait(NTHREADS, extract_data_worker, (
            ofp, peak_cntr, all_peaks, build_predictors, fasta))
    return

main()
