import os, sys

import gzip

import math

import itertools

from collections import namedtuple

import pandas as pd
import numpy as np
#np.random.seed(0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ( 
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier )
from sklearn.metrics import roc_auc_score, average_precision_score

from cross_validation import (
    ClassificationResult, ClassificationResults, iter_train_validation_splits )

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

def estimate_cross_validated_error(data):
    res = ClassificationResults()
    for train, validation in data.iter_train_validation_data_subsets():
        #train = train.balance_data()
        #validation = validation.balance_data()

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

        result_summary = ClassificationResult(
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
    return res

def load_single_motif_data(fname):
    # load the data into a pandas data frame
    data = pd.read_csv(
        fname, sep="\s+", index_col=0, compression='infer')
    return SingleMotifBindingData(data)

def main():
    data = load_single_motif_data(sys.argv[1])
    print estimate_cross_validated_error(data)

if __name__ == '__main__':
    main()
