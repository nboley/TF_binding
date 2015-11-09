import sys
import os
import gzip
import shutil

import math

import itertools
from collections import namedtuple

import multiprocessing

import numpy as np
np.random.seed(0)
from scipy.stats.mstats import mquantiles
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ( 
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier )

from pysam import Fastafile

from grit.lib.multiprocessing_utils import (
    fork_and_wait, ThreadSafeFile, Counter )

import pyTFbindtools

from pyTFbindtools.peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    getFileHandle,
)

from pyTFbindtools.motif_tools import (
    load_selex_models_from_db, 
    load_pwms_from_db, 
    score_region)

from pyTFbindtools.cross_validation import (
    ClassificationResult, 
    ClassificationResults, 
    TrainValidationSplitsThreadSafeIterator,
    find_optimal_ambiguous_peak_threshold )

NTHREADS = 1

class TFBindingData(object):
    @property
    def columns(self):
        return self.data.columns

    @staticmethod
    def merge_labels(labels):
        new_labels = np.zeros(len(labels), dtype=int)
        for i, x in enumerate(labels):
            if isinstance(x, int):
                val = x
            else:
                val = int(x.split(",")[0])
            assert val in (-1, 0, 1)
            new_labels[i] = val
        return new_labels

    def subset_data(self, sample_names, chrs):
        """Return a subsetted copy of self. 
        
        """
        prefixes = set("%s_%s" % (x,y) for (x,y) 
                       in itertools.product(sample_names, chrs))
        indices = [i for i, row in enumerate(self.data.index) 
                   if "_".join(row.split("_")[:2]) in prefixes]
        return type(self)(self.data.iloc[indices].copy())

    def iter_train_validation_data_subsets(self):
        fold_iterator = TrainValidationSplitsThreadSafeIterator(
                self.sample_ids, self.contigs)
        return (
            (self.subset_data(*train_indices), self.subset_data(*val_indices))
            for train_indices, val_indices in fold_iterator 
        )
    
    def _initialize_metadata(self):
        """Caches meta data (after self.data has been set)

        """
        self.label_columns = [
            x for x in self.data.columns if x.startswith('label')]
        self.score_columns = [
            x for x in self.data.columns if x.startswith('score')]
        self.motif_ids = [
            x.split("__")[1] for x in self.label_columns]
        sample_ids = set()
        contigs = set()
        for row_label in self.data.index:
            label_data = row_label.split("_")
            sample_ids.add(label_data[0])
            contigs.add(label_data[1])
        self.sample_ids = sorted(sample_ids)
        self.contigs = sorted(contigs)
        
        return
    
    def __init__(self, pandas_dataframe):
        self.data = pandas_dataframe
        self._initialize_metadata()
        # merge labels for factors with multiple Chipseq experiments in the 
        # same sample
        #for label_column in self.label_columns:
        #    new_labels = self.merge_labels(self.data[label_column])
        #    self.data.loc[:,label_column] = new_labels
        return

class SingleMotifBindingData(TFBindingData):
    def remove_zero_labeled_entries(self):
        return type(self)(self.data.iloc[self.labels != 0,:])

    def balance_data(self):
        """Return a copy of self where the number of positive and negative 
           samples is balanced. 
        
        """
        pos_full = self.data[(self.data[self.label_columns] == 1).all(1)]
        neg_full = self.data[(self.data[self.label_columns] == -1).all(1)]
        sample_size = min(pos_full.shape[0], neg_full.shape[0])
        pos = pos_full.loc[
            np.random.choice(pos_full.index, sample_size, replace=False)]
        neg = neg_full.loc[
            np.random.choice(neg_full.index, sample_size, replace=False)]
        return SingleMotifBindingData( pos.append(neg) )
    
    def __init__(self, pandas_dataframe):
        super(SingleMotifBindingData, self).__init__(pandas_dataframe)
        # make sure this is actually single motif data
        assert len(self.motif_ids) == 1
        assert len(self.label_columns) == 1

    @property
    def labels(self):
        return np.array(self.data[self.label_columns[0]])

class BindingModel():
    def __init__(self):
        #clf = DecisionTreeClassifier(max_depth=20)
        self.clf = RandomForestClassifier(max_depth=10)
        #clf = GradientBoostingClassifier(n_estimators=100)

    def train(self, data):
        self.predictors = [ x for x in data.columns
                            if not x.startswith('label')
                            and not x.startswith('score')]
        assert len(data.label_columns) == 1
        self.label = data.label_columns[0]
        self.score = data.score_columns[0]
        self.train_data = data
        self.mo = self.clf.fit(
            data.data[self.predictors], data.data[self.label])
        return

    def predict(self, predictors):
        return self.mo.predict(predictors)

    def predict_proba(self, predictors):
        return self.mo.predict_proba(predictors)[:,1]
    
    def classify_ambiguous_peaks(self, validation_data, num_thresh=20):
        # find the threhsold that optimizes the F1 score
        best_thresh = find_optimal_ambiguous_peak_threshold(
            self, 
            validation_data.data[self.predictors], 
            validation_data.data[self.label],
            validation_data.data[self.score],
            num_thresh)
        # set the ambiguous peak labels 
        ambiguous_peaks = (validation_data.data[self.label] == 0)
        validation_data.data[self.label][ambiguous_peaks] = 1.0
        ambig_peaks_below_threshold = (
            ambiguous_peaks&(validation_data.data[self.score] <= best_thresh))
        validation_data.data[self.label][ambig_peaks_below_threshold] = -1

        return best_thresh
        
    def evaluate(self, predictors, labels):
        y_hat = self.predict(predictors)
        y_hat_prbs = self.predict_proba(predictors)
        return ClassificationResult(
            labels, y_hat, y_hat_prbs)

def evaluate_model_worker(train_validation_iterator,
                          balance_data,
                          validate_on_clean_labels, 
                          train_on_clean_labels,
                          threadsafe_results_list):
    for train, validation in train_validation_iterator:
        if balance_data:
            train = train.balance_data()
            validation = validation.balance_data()

        if train_on_clean_labels:
            train = train.remove_zero_labeled_entries()

        if validate_on_clean_labels:
            validation = validation.remove_zero_labeled_entries()

        mo = BindingModel()
        mo.train(train)
        mo.classify_ambiguous_peaks(validation)
        res = mo.evaluate(
            validation.data[mo.predictors], validation.data[mo.label])
        pyTFbindtools.log(str(res), 'DEBUG')
        threadsafe_results_list.append(res)

    return

def estimate_cross_validated_error(
        data, 
        balance_data=False, 
        validate_on_clean_labels=False,
        train_on_clean_labels=True):
    from multiprocessing import Manager
    manager = Manager()
    thread_safe_results = manager.list()
    fork_and_wait(NTHREADS, evaluate_model_worker, [
        data.iter_train_validation_data_subsets(),
        balance_data,
        validate_on_clean_labels,
        train_on_clean_labels, 
        thread_safe_results
    ])

    # move the manager data into the results data structure, and close
    # the manager
    res = ClassificationResults(
        thread_safe_results.pop() for x in xrange(len(thread_safe_results)))
    manager.shutdown()
    return res

def load_single_motif_data(fname):
    # load the data into a pandas data frame
    data = pd.read_csv(
        fname, sep="\s+", index_col=0, compression='infer')
    return SingleMotifBindingData(data)

    
class BuildPredictorsFactory(object):
    def build_header(self):
        assert len(self.motifs) == 1
        header = ['region',] + [
            "label__%s" % motif.motif_id for motif in self.motifs] +[
                "score__%s" % motif.motif_id for motif in self.motifs]+ [
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
    fasta = Fastafile(fasta.filename)
    while True:
        index = peak_cntr.return_and_increment()
        if index >= len(peaks): break
        labeled_peak = peaks[index]
        try: 
            scores = build_predictors.build_summary_stats(
                labeled_peak.peak, fasta)
        except Exception, inst: 
            pyTFbindtools.log("ERROR" + str(inst))
            continue
        if index%50000 == 0:
            pyTFbindtools.log("Processed %i/%i peaks" % (index, len(peaks)), 
                              'VERBOSE')
        ofp.write("%s_%s\t%s\t%.4f\t%.4f\t%s\n" % (
            labeled_peak.sample, 
            "_".join(str(x) for x in labeled_peak.peak).ljust(30), 
            labeled_peak.label, 
            labeled_peak.score,
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

    parser.add_argument( '--balance-data', default=False, action='store_true', 
        help='Predict results on balanced labels')
    parser.add_argument( '--skip-ambiguous-peaks', 
        default=False, action='store_true', 
        help='If set, only validate on the clean peak set')

    parser.add_argument( '--half-peak-width', type=int, default=500,
        help='Example accessible region peaks to be +/- --half-peak-width bases from the summit (default: 500)')
    parser.add_argument( '--ofprefix', type=str, default='peakscores',
        help='Output file prefix (default peakscores)')

    parser.add_argument( '--max-num-accessible-regions-per-sample', type=int, 
        help='Choose the top --max-num-accessible-regions-per-sample accessible regions for each sample')
    parser.add_argument( '--max-num-unordered-accessible-regions-per-sample', type=int, 
        help='Randomly choose --max-num-peaks-per-sample peaks to classify for each sample (used for debugging)')

    parser.add_argument( 
        '--verbose', default=False, action='store_true') 
    parser.add_argument( 
        '--debug-verbose', default=False, action='store_true') 

    parser.add_argument( '--threads', '-t', default=1, type=int, 
        help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads
    pyTFbindtools.VERBOSE = (args.verbose or args.debug_verbose)
    pyTFbindtools.DEBUG = (args.debug_verbose)

    annotation_id = 1

    assert ( args.max_num_unordered_accessible_regions_per_sample is None 
             or args.max_num_accessible_regions_per_sample is None), \
        "Doesnt make sense to set both --max-num-accessible-regions-per-sample and --max-num-accessible-regions-per-sample accessible"
    
    if args.selex_motif_id != None:
        motifs = load_selex_models_from_db(motif_ids=[args.selex_motif_id,])
    elif args.cisbp_motif_id != None:
        motifs = load_pwms_from_db(motif_ids=[args.cisbp_motif_id,])
    else:
        assert False, "Must set either --selex-motif-id or --cisbp-motif-id"
    assert len(motifs) == 1
    motif = motifs[0]
    pyTFbindtools.log("Finished loading motifs.", "VERBOSE")

    order_peaks_by_accessibility = False
    num_peaks_per_sample = None
    if args.max_num_unordered_accessible_regions_per_sample is not None:
        assert args.max_num_accessible_regions_per_sample is None
        num_peaks_per_sample = args.max_num_unordered_accessible_regions_per_sample
        order_peaks_by_accessibility = False
    elif args.max_num_accessible_regions_per_sample is not None:
        assert args.max_num_unordered_accessible_regions_per_sample is None
        num_peaks_per_sample = args.max_num_accessible_regions_per_sample
        order_peaks_by_accessibility = True
        
    ofname = "{prefix}.{motif_id}.{half_peak_width}.{max_peaks_per_sample}.{order_peaks}.txt.gz".format(
        prefix=args.ofprefix, 
        motif_id=motifs[0].motif_id,
        half_peak_width=args.half_peak_width,
        max_peaks_per_sample=num_peaks_per_sample,
        order_peaks=order_peaks_by_accessibility
    )
    
    return (annotation_id, motif, ofname,
            args.half_peak_width,
            args.balance_data, 
            args.skip_ambiguous_peaks,
            num_peaks_per_sample, order_peaks_by_accessibility)

def open_or_create_feature_file(
        annotation_id, motif, ofname, 
        half_peak_width,  
        max_n_peaks_per_sample,
        order_by_accessibility):
    try:
        return open(ofname)
    except IOError:
        pyTFbindtools.log("Initializng peaks", "VERBOSE")
        
        pyTFbindtools.log("Creating feature file '%s'" % ofname, 'VERBOSE')
        labeled_peaks = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            motif.tf_id, 
            annotation_id,
            half_peak_width=half_peak_width, 
            max_n_peaks_per_sample=max_n_peaks_per_sample,
            include_ambiguous_peaks=True,
            order_by_accessibility=order_by_accessibility)
        pyTFbindtools.log( "Finished loading peaks.", 'VERBOSE')

        from DB import load_genome_metadata
        fasta_fname = load_genome_metadata(annotation_id).filename
        fasta = Fastafile(fasta_fname)
        pyTFbindtools.log( "Finished initializing fasta.", "VERBOSE")

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
    (annotation_id, motif, ofname, half_peak_width, 
     balance_data, validate_on_clean_labels,
     max_num_peaks_per_sample, order_by_accessibility
        ) = parse_arguments()
    # check to see if this file is cached. If not, create it
    feature_fp = open_or_create_feature_file(
        annotation_id, motif, ofname, 
        half_peak_width=half_peak_width,
        max_n_peaks_per_sample=max_num_peaks_per_sample,
        order_by_accessibility=order_by_accessibility)
    pyTFbindtools.log("Loading feature file '%s'" % ofname, "VERBOSE")
    data = load_single_motif_data(feature_fp.name)
    res = estimate_cross_validated_error(
        data, 
        balance_data=balance_data, 
        validate_on_clean_labels=validate_on_clean_labels,
        train_on_clean_labels=True)
    
    pyTFbindtools.log( str(res) )
    with open(ofname + ".summary", "w") as ofp:
        print >> ofp, res.all_data
    return

if __name__ == '__main__':
    main()
