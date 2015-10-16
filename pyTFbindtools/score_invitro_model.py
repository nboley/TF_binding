import os
import gzip
import shutil

import math

import itertools
from collections import namedtuple

import multiprocessing

import numpy as np
from scipy.stats.mstats import mquantiles
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ( 
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier )
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from pysam import FastaFile, TabixFile

from grit.lib.multiprocessing_utils import (
    fork_and_wait, ThreadSafeFile, Counter )

from peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    getFileHandle )

from motif_tools import (
    load_selex_models_from_db, 
    load_pwms_from_db, 
    score_region)

from cross_validation import (
    ClassificationResult, ClassificationResults, iter_train_validation_splits )


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
        return type(self)(self.data.iloc[indices])

    def iter_train_validation_data_subsets(self):
        for train_indices, val_indices in iter_train_validation_splits(
                self.sample_ids, self.contigs):
            yield ( self.subset_data(*train_indices), 
                    self.subset_data(*val_indices) )
        return
        
    def _initialize_metadata(self):
        """Caches meta data (after self.data has been set)

        """
        self.label_columns = [
            x for x in self.data.columns if x.startswith('label')]
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

def estimate_cross_validated_error(
        data, 
        balance_data=False, 
        validate_on_clean_labels=False,
        train_on_clean_labels=True):
    res = ClassificationResults()
    for train, validation in data.iter_train_validation_data_subsets():
        if balance_data:
            train = train.balance_data()
            validation = validation.balance_data()

        if train_on_clean_labels:
            train = train.remove_zero_labeled_entries()
        
        if validate_on_clean_labels:
            validation = validation.remove_zero_labeled_entries()
        
        #clf_1 = DecisionTreeClassifier(max_depth=20)
        clf_1 = RandomForestClassifier(max_depth=10)
        #clf_1 = GradientBoostingClassifier(n_estimators=100)

        all_predictors = [ x for x in train.columns
                           if not x.startswith('label') ]
        predictors = all_predictors
        
        assert len(train.label_columns) == 1
        label = train.label_columns[0]
        
        mo = clf_1.fit(train.data[predictors], train.data[label])
        y_hat = mo.predict(validation.data[predictors])
        y_hat_prbs = mo.predict_proba(validation.data[predictors])[:,1]
        # set the positives to include real positives and ambiguous positives
        positives = np.array(validation.data[label] > -1)
        num_true_positives = (y_hat[positives] == 1).sum()

        negatives = np.array(validation.data[label] == -1)
        num_true_negatives = (y_hat[negatives] == -1).sum()

        precision, recall, _ = precision_recall_curve(positives, y_hat_prbs)
        prc = np.array([recall,precision])
        auPRC = auc(recall, precision)

        result_summary = ClassificationResult(
            set(train.sample_ids) != set(validation.sample_ids),
            'validation',

            train.contigs, train.sample_ids,

            validation.contigs, validation.sample_ids,

            roc_auc_score(positives, y_hat_prbs),
            auPRC,
            f1_score(positives, y_hat),

            num_true_positives, positives.sum(),
            num_true_negatives, negatives.sum()
        )
        print result_summary
        res.append(result_summary)
    return res

def load_single_motif_data(fname):
    # load the data into a pandas data frame
    data = pd.read_csv(
        fname, sep="\s+", index_col=0, compression='infer')
    return SingleMotifBindingData(data)

    
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

    parser.add_argument( '--balance-data', default=False, action='store_true', 
        help='Predict results on balanced labels')
    parser.add_argument( '--skip-ambiguous-peaks', 
        default=False, action='store_true', 
        help='If set, only validate on the clean peak set')

    parser.add_argument( '--half-peak-width', type=int, default=500,
        help='Example accessible region peaks to be +/- --half-peak-width bases from the summit (default: 500)')
    parser.add_argument( '--ofprefix', type=str, default='peakscores',
        help='Output file prefix (default peakscores)')

    parser.add_argument( '--max-num-peaks-per-sample', type=int, 
        help='the maximum number of peaks to parse for each sample (used for debugging)')

    parser.add_argument( '--threads', '-t', default=1, type=int, 
        help='The number of threads to run.')

    args = parser.parse_args()
    global NTHREADS
    NTHREADS = args.threads

    annotation_id = 1

    if args.selex_motif_id != None:
        motifs = load_selex_models_from_db(motif_ids=[args.selex_motif_id,])
    elif args.cisbp_motif_id != None:
        motifs = load_pwms_from_db(motif_ids=[args.cisbp_motif_id,])
    else:
        assert False, "Must set either --selex-motif-id or --cisbp-motif-id"
    assert len(motifs) == 1
    motif = motifs[0]
    print "Finished loading motifs."

    ofname = "{prefix}.{motif_id}.{half_peak_width}.{max_peaks_per_sample}.txt.gz".format(
        prefix=args.ofprefix, 
        motif_id=motifs[0].motif_id,
        half_peak_width=args.half_peak_width,
        max_peaks_per_sample=args.max_num_peaks_per_sample,
    )
    
    return (annotation_id, motif, ofname,
            args.half_peak_width,
            args.balance_data, 
            args.skip_ambiguous_peaks,
            args.max_num_peaks_per_sample)

def open_or_create_feature_file(
        annotation_id, motif, ofname, 
        half_peak_width,  
        max_n_peaks_per_sample=None):
    try:
        return open(ofname)
    except IOError:
        print "Initializng peaks"
        
        print "Creating feature file '%s'" % ofname
        labeled_peaks = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            motif.tf_id, 
            annotation_id,
            half_peak_width=half_peak_width, 
            max_n_peaks_per_sample=max_n_peaks_per_sample,
            include_ambiguous_peaks=True)
        print "Finished loading peaks."

        from DB import load_genome_metadata
        fasta_fname = load_genome_metadata(annotation_id).filename
        fasta = FastaFile(fasta_fname)
        print "Finished initializing fasta."

        
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
     max_num_peaks_per_sample
        ) = parse_arguments()
    # check to see if this file is cached. If not, create it
    feature_fp = open_or_create_feature_file(
        annotation_id, motif, ofname, 
        half_peak_width=half_peak_width,
        max_n_peaks_per_sample=max_num_peaks_per_sample)
    print "Loading feature file '%s'" % ofname
    data = load_single_motif_data(feature_fp.name)
    res = estimate_cross_validated_error(
        data, 
        balance_data=balance_data, 
        validate_on_clean_labels=validate_on_clean_labels,
        train_on_clean_labels=True)
    
    print res
    with open(ofname + ".summary", "w") as ofp:
        print >> ofp, res.all_data
    return

if __name__ == '__main__':
    main()
