import itertools
from collections import namedtuple, defaultdict

import numpy as np

from scipy.stats import spearmanr, rankdata
from scipy.stats.mstats import mquantiles

from sklearn import cross_validation
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from grit.lib.multiprocessing_utils import Counter

TEST_CHRS = [1,2]
SINGLE_FOLD_VALIDATION_CHRS = range(3,5)
SINGLE_FOLD_TRAIN_CHRS = range(5, 23)

def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1- precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]

ClassificationResultData = namedtuple('ClassificationResult', [
    'is_cross_celltype',
    'sample_type', # should be validation or test
    'train_chromosomes',
    'train_samples', 

    'validation_chromosomes',
    'validation_samples', 

    'auROC', 'auPRC', 'F1', 
    'recall_at_25_fdr', 'recall_at_10_fdr', 'recall_at_05_fdr',
    'num_true_positives', 'num_positives',
    'num_true_negatives', 'num_negatives'])

class ClassificationResult(object):
    _fields = ClassificationResultData._fields

    def __iter__(self):
        return iter(getattr(self, field) for field in self._fields)

    def iter_items(self):
        return zip(self._fields, iter(getattr(self, field) for field in self._fields))
    
    def __init__(self, labels, predicted_labels, predicted_prbs,
                 is_cross_celltype=None, sample_type=None,
                 train_chromosomes=None, train_samples=None,
                 validation_chromosomes=None, validation_samples=None):
        # filter out ambiguous labels
        index = labels > -0.5
        predicted_labels = predicted_labels[index]
        predicted_prbs = predicted_prbs[index]
        labels = labels[index]

        self.is_cross_celltype = is_cross_celltype
        self.sample_type = sample_type

        self.train_chromosomes = train_chromosomes
        self.train_samples = train_samples

        self.validation_chromosomes = validation_chromosomes
        self.validation_samples = validation_samples
        
        positives = np.array(labels == 1)
        self.num_true_positives = (predicted_labels[positives] == 1).sum()
        self.num_positives = positives.sum()
        
        negatives = np.array(labels == 0)        
        self.num_true_negatives = (predicted_labels[negatives] == 0).sum()
        self.num_negatives = negatives.sum()

        if positives.sum() + negatives.sum() < len(labels):
            raise ValueError, "All labels must be either 0 or +1"
        
        try: self.auROC = roc_auc_score(positives, predicted_prbs)
        except ValueError: self.auROC = float('NaN')
        precision, recall, _ = precision_recall_curve(positives, predicted_prbs)
        prc = np.array([recall,precision])
        self.auPRC = auc(recall, precision)
        self.F1 = f1_score(positives, predicted_labels)
        self.recall_at_50_fdr = recall_at_fdr(labels, predicted_prbs, fdr_cutoff=0.50)
        self.recall_at_25_fdr = recall_at_fdr(labels, predicted_prbs, fdr_cutoff=0.25)
        self.recall_at_10_fdr = recall_at_fdr(labels, predicted_prbs, fdr_cutoff=0.10)
        self.recall_at_05_fdr = recall_at_fdr(labels, predicted_prbs, fdr_cutoff=0.05)

        return

    @property
    def positive_accuracy(self):
        return float(self.num_true_positives)/(1e-6 + self.num_positives)

    @property
    def negative_accuracy(self):
        return float(self.num_true_negatives)/(1e-6 + self.num_negatives)

    @property
    def balanced_accuracy(self):
        return (self.positive_accuracy + self.negative_accuracy)/2    

    def iter_numerical_results(self):
        for key, val in self.iter_items():
            try: _ = float(val) 
            except TypeError: continue
            yield key, val
        return

    def __str__(self):
        rv = []
        if self.train_samples is not None:
            rv.append("Train Samples: %s\n" % self.train_samples)
        if self.train_chromosomes is not None:
            rv.append("Train Chromosomes: %s\n" % self.train_chromosomes)
        if self.validation_samples is not None:
            rv.append("Validation Samples: %s\n" % self.validation_samples)
        if self.validation_chromosomes is not None:
            rv.append("Validation Chromosomes: %s\n" % self.validation_chromosomes)
        rv.append("Bal Acc: %.3f" % self.balanced_accuracy )
        rv.append("auROC: %.3f" % self.auROC)
        rv.append("auPRC: %.3f" % self.auPRC)
        rv.append("F1: %.3f" % self.F1)
        rv.append("Re@0.50 FDR: %.3f" % self.recall_at_50_fdr)
        rv.append("Re@0.25 FDR: %.3f" % self.recall_at_25_fdr)
        rv.append("Re@0.10 FDR: %.3f" % self.recall_at_10_fdr)
        rv.append("Re@0.05 FDR: %.3f" % self.recall_at_05_fdr)
        rv.append("Positive Accuracy: %.3f (%i/%i)" % (
            self.positive_accuracy, self.num_true_positives,self.num_positives))
        rv.append("Negative Accuracy: %.3f (%i/%i)" % (
            self.negative_accuracy, self.num_true_negatives, self.num_negatives))
        return "\t".join(rv)

def find_optimal_ambiguous_peak_threshold(
        mo, predictors, labels, peak_scores, num_thresh):
    """Find the threshold that maximizes the F1 score.

    """
    # make a copy of the original ambiguous label set so that we can
    # evaluate the labels at various thresholds 
    original_labels = labels
    labels = labels.copy()
    # find the peaks with ambiguous labels and their scores
    ambiguous_peaks = np.array(original_labels == -1)
    ambiguous_peak_scores = peak_scores[ambiguous_peaks]

    # predict the label proabilities for every peak (ambiguous included)
    y_hat_prbs = mo.predict_proba(predictors)
    # determine the list of thresholds
    ambiguous_thresholds = mquantiles(
        ambiguous_peak_scores, 
        np.arange(0.0,1.0,1.0/(num_thresh-1))).tolist() + [1.0,]
    # set all of the ambiguous peaks' labels to positive. This will change
    # as we try different thresholds
    labels[ambiguous_peaks] = 1.0
    # for each threshold, change the labels and evaluate the model's 
    # performance
    best_f1 = 0.0
    best_thresh = None
    for i, thresh in enumerate(ambiguous_thresholds):
        # "Testing thresh %i/%i" % (i+1, len(ambiguous_thresholds))
        ambig_peaks_below_threshold = (
            ambiguous_peaks&(peak_scores <= thresh))
        labels[ambig_peaks_below_threshold] = 0
        res = mo.evaluate(predictors, labels)
        if res.F1 > best_f1:
            best_f1 = res.F1
            best_thresh = thresh

    return best_thresh

def plot_ambiguous_peaks(scores, pred_prbs, ofname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot

    def make_boxplot(n_groups=20):
        groups = defaultdict(list)
        for i, (score, prb) in enumerate(sorted(zip(scores, pred_prbs))):
            groups[(i*n_groups)/len(pred_prbs)].append(float(prb))
        group_labels = sorted(int((x+0.5)*len(pred_prbs)/n_groups) 
                              for x in groups.keys())
        groups = [x[1] for x in sorted(groups.items())]
        
        matplotlib.pyplot.title("Sample Rank vs Peak Score")
        matplotlib.pyplot.axis([0, len(pred_prbs), -0.01, 1.01])
        matplotlib.pyplot.xlabel("Peak Rank")
        matplotlib.pyplot.ylabel("Label")
        matplotlib.pyplot.boxplot(groups, sym="")    
        matplotlib.pyplot.xticks(
            range(1,n_groups+1), group_labels, rotation='vertical')

    fig = matplotlib.pyplot.figure(num=None, figsize=(10, 7.5))
    make_boxplot()
    matplotlib.pyplot.savefig(ofname)
    return

class ClassificationResults(list):
    def __str__(self):
        if len(self)==0:
            return ''
        balanced_accuracies = [x.balanced_accuracy for x in self]    
        auROCs = [x.auROC for x in self]
        auRPCs = [x.auPRC for x in self]
        f1s = [x.F1 for x in self]
        rv = []
        rv.append("Balanced Accuracies: %.3f (%.3f-%.3f)" % (
            sum(balanced_accuracies)/len(self),
            min(balanced_accuracies), max(balanced_accuracies)) )
        rv.append("auROC:               %.3f (%.3f-%.3f)" % (
            sum(auROCs)/len(self), min(auROCs), max(auROCs)))
        rv.append("auPRC:               %.3f (%.3f-%.3f)" % (
            sum(auRPCs)/len(self), min(auRPCs), max(auRPCs)))
        rv.append("F1:                  %.3f (%.3f-%.3f)" % (
            sum(f1s)/len(self), min(f1s), max(f1s)))
        return "\n".join(rv)

    @property
    def all_data(self):
        # write the header
        rv = []
        rv.append( "\t".join(ClassificationResult._fields))
        for entry in self:
            rv.append("\t".join(str(x) for x in entry))
        return "\n".join(rv)

class TrainValidationSplitsThreadSafeIterator(object):
    def __init__(self, sample_ids, contigs):
        self.tr_valid_indices = list(iter_train_validation_splits(
                sample_ids, contigs))
        self.i = Counter()
        self.n = len(self.tr_valid_indices)

    def __iter__(self):
        return self

    def next(self):
        i = self.i.return_and_increment()
        if i < self.n:
            train_indices, valid_indices = self.tr_valid_indices[i]
            return train_indices, valid_indices
        else:
            raise StopIteration()

def iter_train_validation_splits(sample_ids, contigs,
                                 validation_contigs=None,
                                 single_celltype=False):
    # determine the training and validation sets
    if len(sample_ids) == 1:
        train_samples = sample_ids
        validation_samples = sample_ids
        all_sample_folds = [(train_samples, validation_samples),]
    else:
        all_sample_folds = []
        for sample in sample_ids:
            if single_celltype:
                all_sample_folds.append(([sample,], [sample,]))
            else:
                all_sample_folds.append(
                    ([x for x in sample_ids if x != sample], [sample,]))
    # split the samples into validation and training
    non_test_chrs = sorted(
        set(contigs) - set("chr%i" % i for i in TEST_CHRS))
    if validation_contigs is None:
        all_chr_folds = list(cross_validation.KFold(
            len(non_test_chrs), n_folds=5))
        for sample_fold, chr_fold in itertools.product(
                all_sample_folds, all_chr_folds):
            train_samples, validation_samples = sample_fold
            train_chrs = [non_test_chrs[i] for i in chr_fold[0]]
            validation_chrs = [non_test_chrs[i] for i in chr_fold[1]]
            yield (
                (train_samples, train_chrs), 
                (validation_samples, validation_chrs))
    else: # use provided validation contigs
        train_chrs = set(non_test_chrs) - validation_contigs
        for train_samples, validation_samples in all_sample_folds:
            yield (
                (train_samples, train_chrs),
                (validation_samples, validation_contigs))
    return
