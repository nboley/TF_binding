#!/usr/bin/env python

import os, sys
import time

import math

import cPickle as pickle

from collections import defaultdict
from itertools import chain

from scipy.stats import spearmanr, rankdata

import numpy

from pysam import FastaFile

import multiprocessing.queues
import Queue

import gzip

from grit.files.reads import Reads, MergedReads
from grit.lib.multiprocessing_utils import fork_and_wait, ProcessSafeOPStream
from grit.frag_len import build_normal_density

import pandas as pd
from pandas.tools.plotting import scatter_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from DNABindingProteins import ChIPSeqReads

from motif_tools import estimate_unbnd_conc_in_region, Motif, logistic, R, T

NTHREADS = 1
PLOT = False
MAX_N_PEAKS = 10000

MAX_ENERGY_WIGGLE = -math.log(1e-12)

class ATACSeqReads(Reads):
    pass

def load_peak_region(fasta, contig, start, stop, 
                     atacseq_reads, histone_mark_reads,
                     factors_and_motifs,
                     factors_and_chipseq_reads, frag_len):

    pickle_fname = PeakRegion.build_pickle_fname(
        contig, start, stop, 
        factors_and_chipseq_reads.keys(),
        [x.name for x in chain(*factors_and_motifs.itervalues())])
    try:
        with open(pickle_fname, 'rb') as fp:
            rv = pickle.load(fp)
            return rv
    except IOError:
        pass

    peak = PeakRegion(
        fasta, contig, start, stop)
    peak.add_atacseq_cov(atacseq_reads)
    for factor, motifs in factors_and_motifs.iteritems():
        for motif in motifs:
            peak.add_motif(motif)
    for factor, reads in factors_and_chipseq_reads.iteritems(): 
        peak.add_chipseq_reads(reads)

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

        self.chipseq_cov = defaultdict(dict)
        self.smooth_chipseq_cov = defaultdict(dict)
        
        self.motifs = {} 
        self.pwm_cov = {}
        self.score_cov = {}
    
    def __len__(self):
        return self.stop - self.start
    
    @staticmethod
    def build_str(contig, start, stop, chipseq_factors, motif_names):
        rv = []
        rv.append("%s_%i_%i" % (contig, start, stop))
        rv.append("-".join(sorted(chipseq_factors)))
        rv.append("-".join(sorted(motif_names)))
        return str(hash(".".join(rv)))

    @staticmethod
    def build_pickle_fname(contig, start, stop, chipseq_factors, motif_names):
        return "./CACHED_OBJECTS/.PICKLEDREGION.%s.obj" % PeakRegion.build_str(
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
    
    def add_chipseq_reads(self, chipseq_reads):
        #assert factor not in self.chipseq_cov
        #assert factor not in self.smooth_chipseq_cov
        if isinstance(chipseq_reads, MergedReads):
            reads = chipseq_reads._reads
        else:
            reads = [chipseq_reads,]
        for chipseq_reads in reads:
            rd_cov = chipseq_reads.build_unpaired_reads_fragment_coverage_array(
                self.contig, '.', self.start, self.stop, 1, 15)
            self.chipseq_cov[chipseq_reads.factor][chipseq_reads.id] = rd_cov

            smooth_rd_cov = chipseq_reads.build_unpaired_reads_fragment_coverage_array(
                self.contig, '.', self.start, self.stop, 15)
            self.smooth_chipseq_cov[chipseq_reads.factor][chipseq_reads.id]=( 
                smooth_rd_cov/(smooth_rd_cov.sum()+1e-6))
        return

    def estimate_unbnd_conc_in_region(self, motif_name):
        motif = self.motifs[motif_name]
        score_cov = self.score_cov[motif_name]
        atacseq_cov = self.atacseq_cov
        rd_cov = numpy.zeros(len(self)+1, dtype=float)
        for bsid, cov in self.chipseq_cov[motif.factor].iteritems():
            rd_cov += cov
        frag_len = 125
            
        log_tf_conc = estimate_unbnd_conc_in_region(
            motif, score_cov, atacseq_cov, rd_cov,
            frag_len, MAX_ENERGY_WIGGLE)
        return log_tf_conc

    def get_scores_atac_and_chipseq(self, motif_name):
        motif = self.motifs[motif_name]
        score_cov = self.score_cov[motif_name]
        atacseq_cov = self.atacseq_cov
        rd_cov = numpy.zeros(len(self)+1, dtype=float)
        for bsid, cov in self.chipseq_cov[motif.factor].iteritems():
            rd_cov += cov
        return score_cov, atacseq_cov, rd_cov

    @staticmethod
    def iter_upper_rank_means(scores, percentiles):
        sorted_scores = numpy.sort(scores)[::-1]
        for percentile in percentiles:
            yield percentile, float(sorted_scores[
                :int(percentile*len(sorted_scores))+1].mean())

    def calc_occupancy_scores(self, factor, motif, tf_concs=[1e-6,]):
        mean_rvs = []
        max_rvs = []

        trimmed_atacseq_cov = self.atacseq_cov[len(motif)+1:]
        atacseq_weights = trimmed_atacseq_cov/trimmed_atacseq_cov.max()

        for tf_conc in tf_concs:
            log_tf_conc = numpy.log(tf_conc)

            raw_occ = logistic(
                log_tf_conc - self.score_cov[motif.name]/(R*T))
            occ = raw_occ*atacseq_weights
            mean_rv.append(occ.mean())
            max_rv.append(occ.max())
        return mean_rv, max_rvs
    
    def calc_summary_stats(self):
        header = []
        rv = []
        
        # add on the region and atacseq data
        header.append('pk_length')
        rv.append(self.stop - self.start)

        header.append('ATAC_mean')        
        rv.append(self.atacseq_cov.mean())

        header.append('ATAC_max')        
        rv.append(self.atacseq_cov.max())

        # find all factors with motif and chip-seq data
        factors = sorted(set(motif.factor for name, motif 
                            in self.motifs.iteritems()
                        ).intersection(self.chipseq_cov.iterkeys()))

        percentiles = numpy.array(
            [1e-3, 1e-2, 0.02, 0.05, 0.10, 0.25, 0.50])
        
        for factor in sorted(factors):
            for BSID, cov in self.chipseq_cov[factor].iteritems():
                header.append('%s_%s_mean_ChIPseq_cov' % (factor, BSID))
                rv.append(cov.mean())
            for motif_name, motif in sorted(self.motifs.iteritems()):
                # skip motifs that aren't the correct factor
                if factor != motif.factor: continue

                header.append('%s_mean_score' % motif_name)
                rv.append(self.score_cov[motif_name].mean())

                header.append('%s_max_score' % motif_name)
                rv.append(self.score_cov[motif_name].min())
                #for percentile, score in self.iter_upper_rank_means(
                #        self.score_cov[motif_name], percentiles):
                #    header.append('%s_q_%.2f_score' % (motif_name, percentile))
                #    rv.append(score)

                trimmed_atacseq_cov = self.atacseq_cov[len(motif)+1:]
                atacseq_weights = trimmed_atacseq_cov/trimmed_atacseq_cov.max()
                #1000, trimmed_atacseq_cov.max())
                
                w_pwm_scores = self.pwm_cov[motif_name]*atacseq_weights
                header.append('%s_mean_w_pwm_score' % motif_name)
                rv.append(w_pwm_scores.mean())
                #for percentile, score in self.iter_upper_rank_means(
                #        w_pwm_scores, percentiles):
                #    header.append('%s_q_%.2f_w_pwm_score' % (motif_name, percentile))
                #    rv.append(score)
                    
                #header.append('%s_max_w_pwm_score' % motif_name)
                #rv.append(w_pwm_scores.max())

                """
                for tf_conc in [1e-30, 1e-20, 1e-15, 1e-10, 1e-7, 1e-5, 1e-2, 1e-1, 
                                1, 1e2, 1e5, 1e7, 1e10, 1e15, 1e20, 1e30 ]:
                    log_tf_conc = numpy.log(tf_conc)

                    raw_occ = logistic(
                        log_tf_conc - self.score_cov[motif.name]/(R*T))
                    occ = raw_occ*atacseq_weights
                    header.append('%s_%e_mean_occ' % (motif_name, tf_conc))
                    rv.append(occ.mean())

                    #header.append('%s_%e_max_occ' % (motif_name, tf_conc))
                    #rv.append(occ.max())
                """
                log_tf_conc = numpy.log(1e5)
                raw_occ = logistic(
                    log_tf_conc - self.score_cov[motif_name]/(R*T))
                occ = raw_occ*atacseq_weights
                header.append('%s_mean_occ' % motif_name)
                rv.append(occ.mean())

                header.append('%s_max_occ' % motif_name)
                rv.append(occ.max())

                """
                # XXX
                # find the raw occupancy that provies the best correpondence
                # between the signals, and then try and predict these 
                # sequentially
                unbnd_conc = self.estimate_unbnd_conc_in_region(motif_name)
                #print self.score_cov[motif_name].mean(), unbnd_conc

                #print unbnd_conc
                #unbnd_conc = 0.0
                raw_occ = logistic(
                    unbnd_conc + self.score_cov[motif_name]/(R*T))
                occ = raw_occ*atacseq_weights
                header.append('%s_weighted_occ' % motif_name)
                rv.append(occ.mean())

                header.append('%s_unbnd_conc' % motif_name)
                rv.append(unbnd_conc)
                """

                #for percentile, score in self.iter_upper_rank_means(
                #        occ, percentiles):
                #    header.append('%s_q_%.2f_occ' % (motif_name, percentile))
                #    rv.append(score)

                #header.append('%s_max_occ' % motif_name)
                #rv.append(occ.max())

        return header, rv

def OLD():
    """Just a place to store code temporarily""" 
    pass
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

def process_peaks_worker(proc_queue, ofp, 
                         motif, fasta, 
                         chipseq_reads, atacseq_reads, histone_mark_reads,
                         frag_len):
    # relaod the file handles to make random access process safe 
    fasta = FastaFile(fasta.filename)
    for key, reads in chipseq_reads.iteritems():
        chipseq_reads[key] = reads.reload()
    atacseq_reads = atacseq_reads.reload()
    while proc_queue.qsize() > 0:
        try: 
            region = proc_queue.get(timeout=0.5)[:3]
        except Queue.Empty: 
            continue 
        
        peak = load_peak_region(
            fasta, 
            region[0], max(0, region[1]-2*frag_len), region[2]+2*frag_len, 
            atacseq_reads, histone_mark_reads,
            motif, 
            chipseq_reads, frag_len)

        try: header, vals = peak.calc_summary_stats()
        except:
            print "ERROR"
            continue
        rv = "\t".join(map(str, vals))        
        if proc_queue.qsize() > 0 and proc_queue.qsize() % 100 == 0:
            print >> sys.stderr, "%i\t%i" % (proc_queue.qsize(),os.getpid())
        ofp.write( rv + "\n") 
    
    return

class SummaryResults(pd.DataFrame):
    def get_factors_and_column_indices(self):
        # always include atac seq
        factors_and_columns = defaultdict(list)
        for index, column_name in enumerate(self.columns[2:]):
            factor = column_name.split("_")[0]
            factors_and_columns[factor].append(index+2)
        return dict(factors_and_columns)

    def rank_correlation(self, columns, method='spearman'):
        STEP_SIZE = MAX_N_PEAKS/10
        cuts = range(STEP_SIZE, len(self)+1, STEP_SIZE)
        tf_column_name = min(column for column in self.columns[columns] 
                             if column.endswith('mean_ChIPseq_cov'))
        res = pd.concat(
            [ self[columns][:cut].corr(
                method='spearman')[tf_column_name]
              for cut in cuts ], axis=1 ).transpose()
        res.index = cuts
        return res
    
    def scatter_plot(self, factor):
        f_and_c = self.get_factors_and_column_indices()
        # add in the atac seq column
        columns = [1,] + f_and_c[factor]
        scatter_matrix(self[columns].rank(),  
                       figsize=(16,16), 
                       alpha=0.05, color='black')

    def rank_plot(self, factor):
        f_and_c = self.get_factors_and_column_indices()
        # add in the atac seq column
        columns = [1,] + f_and_c[factor]
        corr_mat = self.rank_correlation(columns)
        corr_mat.plot(figsize=(16,16), table=True, colormap=cm.gist_rainbow)

    

def load_chipseq_reads(all_ChIP_seq_reads):
    factor_grpd_reads = defaultdict(list)
    for fp in all_ChIP_seq_reads:
        reads = ChIPSeqReads(fp.name).init()
        factor_grpd_reads[reads.factor].append(reads)
    
    for factor in factor_grpd_reads.iterkeys():
        factor_grpd_reads[factor] = MergedReads(factor_grpd_reads[factor])
    
    return dict(factor_grpd_reads)

def load_summary_stats(fname):
    data = SummaryResults(pd.read_table(fname))
    for factor in data.get_factors_and_column_indices().iterkeys():
        if PLOT:
            #data.scatter_plot(factor)
            #plt.savefig('%s.%s.scatter.png' % (fname, factor))
            #plt.close()

            data.rank_plot(factor)
            plt.savefig("%s.%s.rankcor.png" % (fname, factor))
            plt.close()
            print "FINISHED ", "%s.%s.rankcor.png" % (fname, factor)


def collect_and_write_peak_summary_stats(
        peaks, motifs, fasta, 
        chipseq_reads, 
        atacseq_reads, histone_mark_reads, 
        frag_len, ofname):
    proc_queue = multiprocessing.Queue()
    for pk in peaks: proc_queue.put(pk)

    # process a single peak so that we know what to name the columns
    region = proc_queue.get()
    peak = load_peak_region(
        fasta, 
        region[0], max(0, region[1]-2*frag_len), region[2]+2*frag_len,
        atacseq_reads, histone_mark_reads,
        motifs, 
        chipseq_reads,
        frag_len)
    header, vals = peak.calc_summary_stats()

    ofp = ProcessSafeOPStream(open(ofname, "w"))
    args = [proc_queue, ofp, motifs, fasta, 
            chipseq_reads, atacseq_reads, histone_mark_reads, frag_len]

    ofp.write("\t".join(header) + "\n")
    ofp.write("\t".join(map(str, vals)) + "\n")

    fork_and_wait(NTHREADS, process_peaks_worker, args)
    # let the printing catch up
    time.sleep(0.1)
    ofp.close()

def collect_and_write_peak_summary_stats(
        peaks, motifs, fasta, 
        chipseq_reads, 
        atacseq_reads, histone_mark_reads, 
        frag_len, ofname):
    proc_queue = multiprocessing.Queue()
    for pk in peaks: proc_queue.put(pk)

    # process a single peak so that we know what to name the columns
    region = proc_queue.get()
    peak = load_peak_region(
        fasta, 
        region[0], max(0, region[1]-2*frag_len), region[2]+2*frag_len,
        atacseq_reads, histone_mark_reads,
        motifs, 
        chipseq_reads,
        frag_len)
    header, vals = peak.calc_summary_stats()

    ofp = ProcessSafeOPStream(open(ofname, "w"))
    args = [proc_queue, ofp, motifs, fasta, 
            chipseq_reads, atacseq_reads, histone_mark_reads, frag_len]

    ofp.write("\t".join(header) + "\n")
    ofp.write("\t".join(map(str, vals)) + "\n")

    fork_and_wait(NTHREADS, process_peaks_worker, args)
    # let the printing catch up
    time.sleep(0.1)
    ofp.close()

def plot_predicted_vs_observed_pks(obs_score, pred_score, ofname):
    plt.figure()
    heatmap, xedges, yedges = numpy.histogram2d(
        rankdata(-obs_score, method='ordinal'), 
        rankdata(pred_score, method='ordinal'), 
        bins=20)
    heatmap, xedges, yedges = numpy.histogram2d(
        numpy.clip(-numpy.log(1+obs_score), -0.1, 0),
        numpy.clip(numpy.log(1+pred_score), 0, 0.1), 
        bins=100)
    #heatmap, xedges, yedges = numpy.histogram2d(
    #    numpy.clip(-numpy.log(1+data['ATAC_mean']), -0.1, 0),
    #    numpy.clip(numpy.log(y), 0, 0.1), 
    #    bins=100)

    #heatmap, xedges, yedges = numpy.histogram2d(
    #    rankdata(-data['ATAC_mean'], method='average'), 
    #    rankdata(obs_score, method='average'), 
    #    bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    #plt.scatter(rankdata(obs_score, method='ordinal'), 
    #            rankdata(pred_score, method='ordinal'))
    plt.savefig(ofname)
    plt.close()
    return

def predict_tf_binding(summary_stats):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn import cross_validation, linear_model
    #rng = numpy.random.RandomState(1)
    #clf_1 = DecisionTreeRegressor(max_depth=20)
    #clf_1 = RandomForestRegressor(n_estimators=100)
    #clf_1 = AdaBoostRegressor(
    #    DecisionTreeRegressor(max_depth=10),
    #    n_estimators=1000, random_state=rng, loss='square')
    #clf_1 = linear_model.LinearRegression()
    clf_1 = GradientBoostingRegressor(loss='lad', n_estimators=250)
    """
    weighted_occ_columns =  ['BATF_known1_weighted_occ', 
                             'BCL11A_disc2_weighted_occ',
                             'BCL11A_disc4_weighted_occ', 
                             'IRF4_2_weighted_occ', 
                             'REST_known1_weighted_occ', 
                             'SPI1_known4_weighted_occ']

    mean_occ_columns =  ['BATF_known1_mean_occ', 
                    'BCL11A_disc2_mean_occ',
                    'BCL11A_disc4_mean_occ', 
                    'IRF4_2_mean_occ', 
                    'REST_known1_mean_occ', 
                    'SPI1_known4_mean_occ']
    
    unbnd_tf_columns = [ 'BATF_known1_unbnd_conc', 
                         'BCL11A_disc2_unbnd_conc',
                         'BCL11A_disc4_unbnd_conc', 
                         'IRF4_2_unbnd_conc', 
                         'REST_known1_unbnd_conc', 
                         'SPI1_known4_unbnd_conc']
    """
    #occ_columns = weighted_occ_columns
    #for col in occ_columns:
    #    summary_stats["OCC_ADJ_" + col ] = summary_stats[col]*summary_stats['ATAC_mean']
    #print summary_stats.head()
    #return
    #print summary_stats.columns

    #print summary_stats.corr(method='spearman')
    
    X = summary_stats[['ATAC_mean',] + [
        x for x in summary_stats.columns if 
        x.endswith('mean_occ') 
        #or x.endswith('mean_score')
        #or x.endswith('max_score')
        or x.endswith('max_occ')
    ] ]

    #X = summary_stats[[ 
    #    x for x in summary_stats.columns if 
    #    x.endswith('mean_occ') 
    #    #or x.endswith('mean_score')
    #] ]
    
    test_indices = numpy.array(range(0,len(X), 5))
    train_indices = numpy.array(sorted(set(xrange(len(X))) - set(test_indices)))
    # group chipseq experiments by factor
    BS1_factors = sorted(c for c in summary_stats.columns 
                         if c.endswith("mean_ChIPseq_cov")
                         and '1_1' in c)
    BS2_factors = sorted(c for c in summary_stats.columns 
                         if c.endswith("mean_ChIPseq_cov")
                         and '2_1' in c)

    for factor in zip(BS1_factors, BS2_factors):
        factor_name = factor[0].split("_")[0]
        for col in X.columns:
            #if not col.startswith(factor_name): continue
            print "Marginal Corr:", factor_name, col, \
                spearmanr(X[col], summary_stats[factor[0]])[0], \
                spearmanr(X[col], summary_stats[factor[1]])[0]

    #assert False
    for bs1_col, bs2_col in zip(BS1_factors, BS2_factors):
        y1 = summary_stats[bs1_col][train_indices]
        clf_1.fit(X.loc[train_indices,:], numpy.log(y1+1))
        y1_hat = clf_1.predict(X.loc[test_indices,:])
        #print clf_1.feature_importances_
        #print clf_1.coef_
        y2 = summary_stats[bs2_col][train_indices]
        clf_1.fit(X.loc[train_indices,:], numpy.log(y2+1))
        y2_hat = clf_1.predict(X.loc[test_indices,:])
        #print clf_1.feature_importances_
        #print clf_1.coef_

        #print ( bs1_col, spearmanr(y1, y2)[0], 
        #        spearmanr(summary_stats.loc[test_indices,bs1_col], y2_hat)[0], 
        #        spearmanr(summary_stats.loc[test_indices,bs2_col], y1_hat)[0] )
        print ( bs1_col, spearmanr(y1, y2)[0], 
                spearmanr(numpy.log(summary_stats.loc[test_indices,bs1_col]+1), y2_hat)[0], 
                spearmanr(numpy.log(summary_stats.loc[test_indices,bs2_col]+1), y1_hat)[0] )

    return

    pass

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate tf binding from sequence and chromatin accessibility data.')

    parser.add_argument( '--fasta', type=file, required=True,
        help='Fasta containing genome sequence.')

    parser.add_argument( '--motifs', type=file,  nargs='+',
        help='Files containing a PWM.')

    parser.add_argument( '--peaks', type=file,  required=True,
        help='Narrowpeak file containing peaks to aggregate over.')

    parser.add_argument( '--ChIP-seq-reads', type=file,  nargs='+',
        help='File containing ChIP-seq reads.')

    parser.add_argument( '--ATAC-seq-reads', type=file,  required=True,
        help='Indexed BAM file containing ATAC seq reads.')

    parser.add_argument( '--Histone-mark-seq-reads', type=file, 
                         nargs='*', default=[], 
        help='Indexed BAM files containing histone mark reads.')

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

    fasta = FastaFile(args.fasta.name)

    atacseq_reads = ATACSeqReads(args.ATAC_seq_reads.name).init(
        True, True, False, False)

    peaks = load_narrow_peaks(args.peaks.name)
    
    # load the motif data
    motifs = defaultdict(list)
    for fp in args.motifs:
        motif = Motif(fp.read().strip())
        motifs[motif.factor].append(motif)
    motifs = dict(motifs)
    
    # load the chipseq data
    chipseq_reads = load_chipseq_reads(args.ChIP_seq_reads)

    # load the chipseq data
    histone_mark_reads = load_chipseq_reads(args.Histone_mark_seq_reads)

    frag_len = 150
    
    return (args.output_fname, 
            motifs, fasta, 
            chipseq_reads, atacseq_reads, histone_mark_reads,
            peaks, frag_len )


def find_optimal_GFE(motif, pks, chipseq_scores, atacseq_signal):
    res = []
    max_len = max(len(pk) for pk in pks)
    scores = numpy.zeros((len(pks), max_len+1))

    for GFE in numpy.arange(-20, 10, 1.0):
        motif.build_occupancy_weights(4, GFE)
        for pk_i, pk in enumerate(pks):
            score_cov = numpy.array(
                [score for pos, score in motif.iter_seq_score(pk.seq)])
            scores[pk_i, len(motif)+1:] = score_cov
        res.append((
            spearmanr((logistic(-scores/(R*T))*atacseq_signal).mean(1), chipseq_scores)[0],
            GFE))
        print motif.name, GFE, res[-1]
    max_cor = max(x[0] for x in res)
    print motif.name, [x[1] for x in res if x[0] == max_cor][0], max_cor
    print >> sys.stderr, motif.name, [x[1] for x in res if x[0] == max_cor][0], max_cor

    return (motif.name, [x[1] for x in res if x[0] == max_cor][0], max_cor)

def main():
    ( ofname, motifs, fasta, 
      chipseq_reads, atacseq_reads, 
      histone_mark_reads,
      peaks, 
      frag_len ) = parse_arguments()

    pk_size = 100
    num_pks = 500
    pks = []
    for i, (contig, start, stop, summit) in enumerate(peaks):
        if i >= num_pks: break
        if i%10 == 0: print i
        if start+summit < pk_size: continue
        pk = load_peak_region(fasta, 
                              contig, start+summit-pk_size, start+summit+pk_size,
                              atacseq_reads, histone_mark_reads,
                              motifs, chipseq_reads, 150)
        pks.append(pk)
    
    max_len = max(len(pk) for pk in pks)
    print "Max Length: ", max_len
    atacseq_signal = numpy.zeros((num_pks, max_len+1))
    for pk_i, pk in enumerate(pks):
        atacseq_signal[pk_i,:] = pk.atacseq_cov
    
    pids = []
    for factor, motifs in motifs.items():
        chipseq_signal = numpy.zeros((num_pks, max_len+1))
        for pk_i, pk in enumerate(pks):
            for bsid, cov in pk.chipseq_cov[factor].iteritems():
                chipseq_signal[pk_i,:] += cov
        chipseq_scores = chipseq_signal.mean(1)
        
        for motif in motifs:
            pid = os.fork()
            if pid == 0:
                res = find_optimal_GFE(motif, pks, chipseq_scores, atacseq_signal)
                print "="*10, res
                print >> sys.stderr, "="*10, res
                os._exit(0)
            else:
                pids.append(pid)
    
    for pid in pids:
        os.waitpid(pid, 0)
    
    return
    
    if True:
        collect_and_write_peak_summary_stats(
            peaks, motifs, fasta, 
            chipseq_reads, 
            atacseq_reads, histone_mark_reads, 
            frag_len, ofname)

    data = SummaryResults(pd.read_table(ofname))
    print "Finished loading data"
    predict_tf_binding(data)


if __name__ == '__main__':
    main()
