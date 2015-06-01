import os, sys
import time

import cPickle as pickle

from collections import defaultdict
import numpy
import math

from scipy.optimize import brute

from pysam import FastaFile

import multiprocessing.queues
import Queue

import gzip

from grit.files.reads import Reads
from grit.lib.multiprocessing_utils import fork_and_wait, ProcessSafeOPStream

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = 300
R = 1.987-3 # in kCal
R = 8.314e-3

REG_LEN = 10000

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RC_base_map = {'A': 3, 'C': 2, 'G': 1, 'T': 0}

NTHREADS = 1
PLOT = False
MAX_N_PEAKS = 25000

def logit(x):
    return math.log(x) - math.log(1-x)

def logistic(x):
    try: e_x = math.exp(x)
    except: e_x = numpy.exp(x)
    return e_x/(1+e_x)

class Motif():
    def __len__(self):
        return self.length

    def iter_pwm_score(self, seq):
        seq = seq.upper()
        for offset in xrange(len(seq) - len(self)):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = 0.0
            RC_score = 0.0
            if 'N' in subseq: 
                yield offset + len(self)/2, 0.25*len(self)
                continue
            for i, base in enumerate(subseq):
                score += self.pwm[i][base_map[base]]
                RC_score += self.pwm[len(self)-i-1][RC_base_map[base]]
            yield offset + len(self)/2, max(score, RC_score)

    def iter_seq_score(self, seq):
        seq = seq.upper()
        for offset in xrange(len(seq) - len(self)):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = self.consensus_energy
            RC_score = self.consensus_energy
            if 'N' in subseq:
                yield offset + len(self)/2, self.mean_energy
                continue
            for i, base in enumerate(subseq):
                score += self.motif_data[i][base_map[base]]
                RC_score += self.motif_data[len(self)-i-1][RC_base_map[base]]
            yield offset + len(self)/2, max(score, RC_score)

    
    def __init__(self, fp):
        # load the motif data
        lines = fp.read().strip().split("\n")
        
        self.name = lines[0].split()[0][1:]
        self.factor = self.name.split("_")[0]
        
        self.length = len(lines)-1

        self.consensus_energy = 0.0
        self.motif_data = numpy.zeros((self.length, 4), dtype=float)
        
        self.pwm = numpy.zeros((self.length, 4), dtype=float)
        
        for i, line in enumerate(lines[1:]):
            row = numpy.array([logit(1e-3/2 + (1-1e-3)*float(x)) 
                               for x in line.split()[1:]])
            max_val = row.max()
            self.consensus_energy += max_val
            row -= max_val
            self.motif_data[i, :] = row

            pwm_row = numpy.array([
                float(x) for x in line.split()[1:]])
            self.pwm[i, :] = pwm_row
        
        # reset the consensus energy so that the strongest binder
        # has a binding occupancy of 0.999 at chemical affinity 1
        self.consensus_energy = logit(0.999)
        
        # calculate the mean binding energy
        self.mean_energy = sum(row.sum()/4 for row in self.motif_data) 
        
        
        scaled_mean_energy = self.mean_energy
        while logistic(self.consensus_energy*R*T)/logistic(
                self.consensus_energy*R*T + scaled_mean_energy*R*T) > 1e12:
            scaled_mean_energy *= 0.95
        self.motif_data /= (self.mean_energy/scaled_mean_energy)
        self.mean_energy = scaled_mean_energy
        
        print >> sys.stderr,  "Consensus Occupancy:", logistic(
            self.consensus_energy*R*T)
        print >> sys.stderr,  "Mean Occupancy:", logistic(
            self.consensus_energy*R*T + scaled_mean_energy*R*T)

        # change the units into KJ 
        self.consensus_energy *= (R*T)
        self.motif_data *= (R*T)

        print >> sys.stderr,  self.factor
        print >> sys.stderr,  self.consensus_energy
        print >> sys.stderr, logistic(self.consensus_energy)
        print >> sys.stderr, self.mean_energy
        print >> sys.stderr, logistic(self.mean_energy)
        print >> sys.stderr, self.motif_data
        #assert False
        return

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

def build_wig(fasta, motif, region):
    chrm, start, stop, summit = region
    
    output = []

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
    
    summit_line = "%i\t%i\t%.2f\t%.2f\n" % (
        best_pos, summit, 
        best_pos/float(stop-start), summit/float(stop-start))

    return output, summit_line

def build_wig_worker(fasta_fname, regions, motif):
    fasta = FastaFile(fasta_fname)
    n_regions = regions.qsize()
    output = []
    output_summits = []
    while not regions.empty():
        try: region = regions.get(timeout=0.1)
        except Queue.Empty: return
        region_output, summit = build_wig(fasta, motif, region)
        output.extend( region_output )
        output_summits.append(summit)
    
    ofp.write("".join(output))
    of_summits.write("".join(output))
    
    return

class Peaks(list):
    pass

def load_narrow_peaks(fname):
    peaks = Peaks()
    with gzip.open(fname) as fp:
        for i, line in enumerate(fp):
            if line.startswith("track"): continue
            if i > MAX_N_PEAKS: break
            data = line.split()
            chrm = data[0]
            start = int(data[1])
            stop = int(data[2])
            summit = int(data[9])
            peaks.append((chrm, start, stop, summit))
            #proc_queue.put((chrm, start+summit-50, start+summit+50, summit))
    return peaks

def process_all_peaks(fasta_fname, proc_queue, binding_model):
    pids = []
    for i in xrange(24):
        pid = os.fork()
        if pid == 0:
            build_wig_worker()
            os._exit(0)
        else:
            pids.append(pid)
    for pid in pids:
        os.wait(pid, 0)

    return

def build_CHiPseq_unpaired_reads_fragment_coverage_array( 
        self, chrm, strand, start, stop, frag_len, binding_site_size ):
    assert stop >= start
    full_region_len = stop - start + 1
    cvg = numpy.zeros(full_region_len)
    for rd, strand in self.iter_reads_and_strand(chrm, start, stop):
        if strand == '+': 
            rd_start = rd.pos # + binding_site_size
            rd_stop = rd.pos + frag_len # + binding_site_size
        elif strand == '-': 
            rd_start = rd.aend - frag_len
            rd_stop = rd.aend
        else:
            assert False
        cvg[max(0, rd_start-start):max(0, rd_stop-start)] += 1.0/(rd_stop-rd_start+1)

    return cvg


def get_region_data( peak, motif, fasta, 
                     chipseq_reads, atacseq_reads,
                     frag_len):
    #atacseq_cov[atacseq_cov < (0.10*atacseq_cov.max())] = 0.0


    return rd_cov, smooth_rd_cov, atacseq_cov, scores, pwm_scores

def load_peak_region(fasta, contig, start, stop, 
                     atacseq_reads, 
                     motif,
                     chipseq_reads, frag_len):
    pickle_fname = PeakRegion.build_pickle_fname(
        contig, start, stop, 
        [motif.factor,],
        [motif.name,])
    try:
        with open(pickle_fname, 'rb') as fp:
            rv = pickle.load(fp)
            return rv
    except IOError:
        pass

    #with open(os.path.join(op_prefix), ".%s.obj" % str(self), "w") as ofp:
    #    pickle.dump(self, ofp)

    peak = PeakRegion(
        fasta, contig, start, stop)
    peak.add_motif(motif)
    peak.add_atacseq_cov(atacseq_reads)
    peak.add_chipseq_experiment(motif.factor, chipseq_reads, frag_len)

    peak.write_to_disk()

    return peak

class PeakRegion(object):
    def __init__(self, fasta, contig, start, stop):
        self.contig = contig
        self.start = start
        self.stop = stop
        
        self.seq = fasta.fetch(contig, start, stop)
        self.control_seq = None
        
        self.atacseq_cov = None

        self.chipseq_cov = {}
        self.smooth_chipseq_cov = {}
        
        self.motifs = {} 
        self.pwm_cov = {}
        self.score_cov = {}

    @staticmethod
    def build_str(contig, start, stop, chipseq_factors, motif_names):
        rv = []
        rv.append("%s_%i_%i" % (contig, start, stop))
        rv.append("-".join(sorted(chipseq_factors)))
        rv.append("-".join(sorted(motif_names)))
        return ".".join(rv)

    @staticmethod
    def build_pickle_fname(contig, start, stop, chipseq_factors, motif_names):
        return ".PICKLEDREGION.%s.obj" % PeakRegion.build_str(
            contig, start, stop, chipseq_factors, motif_names)

    @property
    def pickle_fname(self):
        return self.build_pickle_fname(
            self.contig, self.start, self.stop,
            self.chipseq_cov.iterkeys(),
            self.motifs.iterkeys()
        )

    def __str__(self):
        return self.build_str(
            self.contig, self.start, self.stop,
            self.chipseq_cov.iterkeys(),
            self.motifs.iterkeys()
        )
    
    def write_to_disk(self, op_prefix='.'):
        op_fname = os.path.join(op_prefix, self.pickle_fname)
        with open(op_fname, "w") as ofp:
            pickle.dump(self, ofp)

    def add_atacseq_cov(self, atacseq_reads):
        self.atacseq_cov = atacseq_reads.build_read_coverage_array(
            self.contig, '.', self.start, self.stop)
        return
    
    def add_motif(self, motif):
        self.motifs[motif.name] = motif
        self.score_cov[motif.name] = numpy.array(
            [score for pos, score in motif.iter_seq_score(self.seq)])
        self.pwm_cov[motif.name] = numpy.array(
            [score for pos, score in motif.iter_pwm_score(self.seq)])
        return
    
    def add_chipseq_experiment(self, factor, chipseq_reads, frag_len):
        assert factor not in self.chipseq_cov
        assert factor not in self.smooth_chipseq_cov
        
        rd_cov = build_CHiPseq_unpaired_reads_fragment_coverage_array(
            chipseq_reads, self.contig, '.', self.start, self.stop, 1, 15)
        self.chipseq_cov[factor] = rd_cov

        smooth_rd_cov = build_CHiPseq_unpaired_reads_fragment_coverage_array(
            chipseq_reads,
            self.contig, '.', self.start, self.stop, 
            frag_len, 15)
        self.smooth_chipseq_cov[factor] = (
            smooth_rd_cov/(smooth_rd_cov.sum()+1e-6))
        return

    def calc_summary_stats(self):
        log_tf_conc = 0.0

        header = []
        rv = []
        
        # add on the region and atacseq data
        header.append('pk_length')
        rv.append(self.stop - self.start)

        header.append('ATAC_mean')        
        rv.append(self.atacseq_cov.mean())

        # find all factors with motif and chip-seq data
        factors = sorted(set(motif.factor for name, motif 
                            in self.motifs.iteritems()
                        ).intersection(self.chipseq_cov.iterkeys()))

        for factor in sorted(factors):
            header.append('TF_mean_cov')
            rv.append(self.chipseq_cov[factor].mean())
            
            for motif_name, motif in sorted(self.motifs.iteritems()):
                # skip motifs that aren't the correct factor
                if factor != motif.factor: continue

                header.append('%s_mean_score' % motif_name)
                rv.append(self.score_cov[motif_name].mean())
                #header.append('%s_max_score' % motif_name)
                #rv.append(self.score_cov[motif_name].max())

                trimmed_atacseq_cov = self.atacseq_cov[len(motif)+1:]
                atacseq_weights = trimmed_atacseq_cov/trimmed_atacseq_cov.max()
                
                w_pwm_scores = self.pwm_cov[motif_name]*atacseq_weights
                header.append('%s_mean_w_pwm_score' % motif_name)
                rv.append(w_pwm_scores.mean())
                #header.append('%s_max_w_pwm_score' % motif_name)
                #rv.append(w_pwm_scores.max())
                
                raw_occ = logistic(
                    log_tf_conc + self.score_cov[motif_name]/(R*T))
                occ = raw_occ*atacseq_weights

                header.append('%s_mean_occ' % motif_name)
                rv.append(occ.mean())
                #header.append('%s_max_occ' % motif_name)
                #rv.append(occ.max())

        return header, rv

def calculate_enrichment(
        peak, motif, fasta, 
        chipseq_reads, atacseq_reads, 
        frag_len, sm_window):
    peak = PeakRegion(fasta, peak[0], peak[1], peak[2])
    peak.add_motif(motif)
    peak.add_atacseq_cov(atacseq_reads)
    peak.add_chipseq_experiment(motif.factor, chipseq_reads, frag_len)
    val, header = peak.calc_summary_stats()
    print header
    print val
    assert False

    #return (atacseq_cov.sum(), atacseq_cov.mean(), atacseq_cov.max(),
    #        rd_cov.sum(), rd_cov.mean(), rd_cov.max(), 
    #        scores.mean(), scores.max())

def estimate_unbnd_conc_in_region(
        peak, motif, fasta, 
        chipseq_reads, atacseq_reads, 
        frag_len, sm_window):

    rd_cov, smooth_rd_cov, atacseq_cov, scores = get_region_data(
        peak, motif, fasta, 
        chipseq_reads, atacseq_reads,
        frag_len)
    
    def build_occ(log_tf_conc):
        raw_occ = logistic(log_tf_conc + scores/(R*T))
        #raw_occ = raw_occ/raw_occ.sum()

        occ = raw_occ*(atacseq_cov/atacseq_cov.max())
        #occ = occ/occ.sum()

        smoothed_occ = numpy.convolve(sm_window, occ/occ.sum(), mode='same')

        return raw_occ, occ, smoothed_occ

    def calc_lhd(log_tf_conc):
        raw_occ, occ, smoothed_occ = build_occ(log_tf_conc)
        diff = (100*smoothed_occ - 100*rd_cov/rd_cov.sum())**2
        lhd = -(numpy.log(smoothed_occ + 1e-12)*rd_cov).sum()
        #print log_tf_conc, diff.sum()
        return lhd

    res = brute(calc_lhd, ranges=(slice(-200, 40, 1),))
    log_tf_conc = max(-200, min(40, res))
    raw_occ, occ, smoothed_occ = build_occ(log_tf_conc)
    diff = (smoothed_occ - rd_cov/rd_cov.sum())**2

    genome_occ_ratio = logistic((log_tf_conc+motif.consensus_energy)/(R*T))/logistic(
        (log_tf_conc + motif.consensus_energy + motif.mean_energy)/(R*T))
    region_occ_ratio = logistic((log_tf_conc+motif.consensus_energy)/(R*T))/raw_occ.mean()

    if PLOT:
        smoothed_atac = numpy.convolve(sm_window, atacseq_cov/atacseq_cov.sum(), mode='same')

        plt.plot( xrange(len(diff)), atacseq_cov/atacseq_cov.sum())
        plt.plot( xrange(len(diff)), smoothed_atac/smoothed_atac.sum())
        plt.plot( xrange(len(diff)), occ*(smooth_rd_cov.max()/occ.max()), 'o' )
        plt.plot( xrange(len(diff)), smoothed_occ/smoothed_occ.sum() )
        #plt.plot( xrange(len(diff)), rd_cov )
        plt.plot( xrange(len(diff)), smooth_rd_cov/smooth_rd_cov.sum() )
        plt.savefig("diff_%s_%i.png" % ("-".join(str(x) for x in peak), 100*log_tf_conc))
        plt.close()

    print >> sys.stderr, log_tf_conc, \
        "%.2e\t%e" % (genome_occ_ratio, region_occ_ratio), \
        (diff**2).sum()*1e6, peak
    print >> sys.stderr,  "Region Av Occ:", raw_occ.mean()
    print >> sys.stderr,  "Consensus Occ:", \
        logistic(log_tf_conc+motif.consensus_energy/(R*T)), \
        log_tf_conc+motif.consensus_energy/(R*T)
    return

def process_peaks_worker(proc_queue, ofp, 
                         motif, fasta, 
                         chipseq_reads, atacseq_reads, 
                         frag_len):
    # relaod the file handles to make random access process safe 
    fasta = FastaFile(fasta.filename)
    chipseq_reads = chipseq_reads.reload()
    atacseq_reads = atacseq_reads.reload()
    while proc_queue.qsize() > 0:
        try: 
            region = proc_queue.get(timeout=0.5)[:3]
        except Queue.Empty: 
            continue 
        
        peak = load_peak_region(
            fasta, region[0], region[1]-2*frag_len, region[2]+2*frag_len, 
            atacseq_reads, 
            motif, 
            chipseq_reads, frag_len)

        header, vals = peak.calc_summary_stats()
        rv = "\t".join(map(str, vals))        
        print >> sys.stderr, "%i\t%i\t%s" % (proc_queue.qsize(),os.getpid(),rv)
        ofp.write( rv + "\n") 
    
    return

class SummaryResults(pd.DataFrame):
    def scatter_plot(self, fname):
        from pandas.tools.plotting import scatter_matrix
        scatter_matrix(self[range(1,len(self.columns))].rank(),  
                       figsize=(16,16), 
                       alpha=0.05, color='black')
        plt.savefig(fname)
        plt.close()

    def rank_correlation(self, method='spearman'):
        STEP_SIZE = MAX_N_PEAKS/25
        cuts = range(STEP_SIZE, len(self)+1, STEP_SIZE)
        res = pd.concat(
            [ self[range(1,len(self.columns))][:cut].corr(
                method='spearman')['TF_mean_cov']
              for cut in cuts ], axis=1 ).transpose()
        res.index = cuts
        return res

    def rank_plot(self, fname):
        corr_mat = self.rank_correlation()
        corr_mat.plot(figsize=(16,16))
        plt.savefig(fname)
        plt.close()

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate unbound tf concentration in a region.')

    parser.add_argument( '--fasta', type=file, required=True,
        help='Fasta containing genome sequence.')

    parser.add_argument( '--motif', type=file,  required=True,
        help='File containing PWM.')

    parser.add_argument( '--peaks', type=file,  required=True,
        help='Narrowpeak file containing peaks to aggregate over.')

    parser.add_argument( '--ChIP-seq-reads', type=file,  required=True,
        help='File containing ChIP-seq reads.')

    parser.add_argument( '--ATAC-seq-reads', type=file,  required=True,
        help='Indexed BAM file containing ATAC seq reads.')

    parser.add_argument( '--output-fname', '-o', required=True,
        help='Output filename.')

    parser.add_argument( '--plot', '-p', default=False, 
                         action='store_true',
                         help='Create plots for each region.')
    
    parser.add_argument( '--threads', '-t', default=1, type=int,
                         help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads

    global PLOT
    PLOT = args.plot
    
    # load the motif data
    motif = Motif(args.motif)

    # load the fasta file
    fasta = FastaFile(args.fasta.name)
    
    chipseq_reads = Reads(args.ChIP_seq_reads.name).init(
        False, False, True, False)
    atacseq_reads = Reads(args.ATAC_seq_reads.name).init(
        True, True, False, False)

    peaks = load_narrow_peaks(args.peaks.name)

    frag_len = 150
    
    return (args.output_fname, 
            motif, fasta, 
            chipseq_reads, atacseq_reads, 
            peaks, 
            frag_len)


def load_summary_stats(fname):
    data = SummaryResults(pd.read_table(fname))

    corr_mat = data.rank_correlation()
    
    if PLOT:
        data.scatter_plot('%s.scatter.png' % fname)
        data.rank_plot("%s.rankcor.png" % fname)


def main():
    ( ofname, motif, fasta, 
      chipseq_reads, atacseq_reads, 
      chipseq_peaks, frag_len ) = parse_arguments()


    """
    for region in chipseq_peaks:
        peak = PeakRegion(fasta, region[0], region[1], region[2])
        peak.add_motif(motif)
        peak.add_atacseq_cov(atacseq_reads)
        peak.add_chipseq_experiment(motif.factor, chipseq_reads, frag_len)
        ps = pickle.dumps(peak)
        print len(ps)
        print peak.calc_summary_stats()
        assert False
    """

    """
    sm_window = numpy.ones(frag_len, dtype=float)/frag_len
    sm_window = numpy.bartlett(2*frag_len)
    sm_window = sm_window/sm_window.sum()

    # make plots
    for peak in chipseq_peaks:
        peak = (peak[0], peak[1]-2*frag_len, peak[2]+2*frag_len)
        try: 
            estimate_unbnd_conc_in_region(
                peak, motif, fasta, 
                chipseq_reads, atacseq_reads, 
                frag_len, sm_window)
        except:
            print "ERROR"
    """ 
    
    proc_queue = multiprocessing.Queue()
    for pk in chipseq_peaks: proc_queue.put(pk)

    ofp = ProcessSafeOPStream(open(ofname, "w"))
    args = [proc_queue, ofp, motif, fasta, 
            chipseq_reads, atacseq_reads, frag_len]

    region = proc_queue.get()
    peak = load_peak_region(
        fasta, region[0], region[1]-2*frag_len, region[2]+2*frag_len, 
        atacseq_reads, 
        motif, 
        chipseq_reads, frag_len)
    header, vals = peak.calc_summary_stats()

    ofp.write("\t".join(header) + "\n")
    ofp.write("\t".join(map(str, vals)) + "\n")

    fork_and_wait(NTHREADS, process_peaks_worker, args)
    # let the printing catch up
    time.sleep(0.1)
    ofp.close()
    
    load_summary_stats(ofname)

if __name__ == '__main__':
    main()
