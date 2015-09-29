import os, sys

import gzip

import math

import itertools

from collections import namedtuple

import pandas as pd
import numpy as np
#np.random.seed(0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import cross_validation, linear_model
from sklearn.metrics import roc_auc_score, average_precision_score

TEST_CHRS = [1,2]
JOHNNIES_VALIDATION_CHRS = range(3,5)
JOHNNIES_TRAIN_CHRS = range(5, 23)
USE_JOHNNIES_TEST_TRAIN_SPLIT = False

ClassificatonResultData = namedtuple('ClassificatonResult', [
    'is_cross_celltype',
    'sample_type', # should be validation or test
    'train_chromosomes',
    'train_samples', 

    'validation_chromosomes',
    'validation_samples', 

    'auROC', 'auPRC', 
    'num_true_positives', 'num_positives',
    'num_true_negatives', 'num_negatives'])

class ClassificatonResult(ClassificatonResultData):
    @property
    def positive_accuracy(self):
        return float(self.num_true_positives)/self.num_positives

    @property
    def negative_accuracy(self):
        return float(self.num_true_negatives)/self.num_negatives

    @property
    def balanced_accuracy(self):
        return (self.positive_accuracy + self.negative_accuracy)/2    
    
    def __str__(self):
        rv = []
        #rv.append(str(self.validation_samples).ljust(25))
        #rv.append(str(self.train_samples).ljust(15))
        rv.append("Balanced Accuracy: %.3f" % self.balanced_accuracy )
        rv.append("auROC: %.3f" % self.auROC)
        rv.append("auPRC: %.3f" % self.auPRC)
        rv.append("Positive Accuracy: %.3f (%i/%i)" % (
            self.positive_accuracy, self.num_true_positives,self.num_positives))
        rv.append("Negative Accuracy: %.3f (%i/%i)" % (
            self.negative_accuracy, self.num_true_negatives,self.num_negatives))
        return "\t".join(rv)

class ClassificatonResults(list):
    def __str__(self):
        balanced_accuracies = [x.balanced_accuracy for x in self]    
        auROCs = [x.auROC for x in self]
        auRPCs = [x.auPRC for x in self]
        rv = []
        rv.append("Balanced Accuracies: [%.3f-%.3f]" % (
            min(balanced_accuracies), max(balanced_accuracies)) )
        rv.append("auROC:               [%.3f-%.3f]" % (
            min(auROCs), max(auROCs)))
        rv.append("auPRC:               [%.3f-%.3f]" % (
            min(auRPCs), max(auRPCs)))
        return "\n".join(rv)

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
            new_labels[i] = -1 if val <= 0 else 1
        return new_labels

    def subset_data(self, sample_names, chrs):
        """Return a subsetted copy of self. 
        
        """
        prefixes = set("%s_%s" % (x,y) for (x,y) 
                       in itertools.product(sample_names, chrs))
        indices = [i for i, row in enumerate(self.data.index) 
                   if "_".join(row.split("_")[:2]) in prefixes]
        return type(self)(self.data.iloc[indices])

    def iter_train_validation_splits(self):
        # determine the training and validation sets
        if len(self.sample_ids) == 1:
            train_samples = self.sample_ids
            validation_samples = self.sample_ids
            all_sample_folds = [(train_samples, validation_samples),]
        else:
            all_sample_folds = []
            for sample in self.sample_ids:
                all_sample_folds.append(
                    ([x for x in self.sample_ids if x != sample], [sample,]))
        # split the samples into validation and training
        non_test_chrs = sorted(
            set(self.contigs) - set("chr%i" % i for i in TEST_CHRS))
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
        return
    
    def iter_train_validation_data_subsets(self):
        for train_indices, val_indices in self.iter_train_validation_splits():
            yield ( self.subset_data(*train_indices), 
                    self.subset_data(*val_indices) )
        return
    
    def split_into_training_and_test(self):
        train_chrs = ['chr%i' %i for i in xrange(5, 23)]
        train = self.subset_data(self.train_samples, train_chrs)
        validation_chrs = ['chr%i' %i for i in xrange(3, 5)]
        validation = self.subset_data(
            self.validation_samples, validation_chrs)
        return train, validation
    
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
        for label_column in self.label_columns:
            new_labels = self.merge_labels(self.data[label_column])
            self.data.loc[:,label_column] = new_labels
        return

class SingleMotifBindingData(TFBindingData):
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

def write_split_data_into_bed():
    """

    """
    assert False, "this is just some code that I never cleaned, but dont want to toss"
    for train, validation in data.iter_train_validation_data_subsets():
        data_subset = validation
        for row, label in itertools.izip(
                data_subset.data.index, data_subset.data[data_subset.label_columns[0]]):
            if label != -1: continue
            print "\t".join(row.split("_")[1:4]) + "\t%s" % row.split("_")[0]
        break

def estimate_cross_validated_error(data):
    res = ClassificatonResults()
    for train, validation in data.iter_train_validation_data_subsets():
        train = train.balance_data()
        validation = validation.balance_data()

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
        y_hat_prbs = mo.predict_proba(validation.data[predictors])
        positives = np.array(validation.data[label] == 1)
        num_true_positives = (y_hat[positives] == 1).sum()

        negatives = np.array(validation.data[label] == -1)
        num_true_negatives = (y_hat[negatives] == -1).sum()

        result_summary = ClassificatonResult(
            set(train.sample_ids) != set(validation.sample_ids),
            'validation',

            train.contigs, train.sample_ids,

            validation.contigs, validation.sample_ids,

            roc_auc_score(validation.data[label], y_hat_prbs[:,1]),
            average_precision_score(validation.data[label], y_hat_prbs[:,1]),

            num_true_positives, positives.sum(),
            num_true_negatives, negatives.sum()
        )
        print result_summary
        res.append(result_summary)
    print res
    return res

def load_single_motif_data(fname):
    # load the data into a pandas data frame
    data = pd.read_csv(
        fname, sep="\s+", index_col=0, compression='infer')
    return SingleMotifBindingData(data)

data = load_single_motif_data(sys.argv[1])
estimate_cross_validated_error(data)
