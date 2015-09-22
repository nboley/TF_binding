import os, sys

import gzip

import math

import itertools

import pandas as pd
import numpy as np
#np.random.seed(0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import cross_validation, linear_model

def load_data(fname):
    def subset_data(data, cells, chrs):
        prefixes = set("%s_%s" % (x,y) for (x,y) 
                       in itertools.product(cells, chrs))
        indices = [i for i, row in enumerate(data.index) 
                   if "_".join(row.split("_")[:2]) in prefixes]
        return data.iloc[indices]
        
    def balance_data(data):
        label_columns = [x for x in data.columns if x.startswith('label')]
        if len(label_columns) > 1:
            return data[(data[label_columns] != 0).all(1)]
        else:
            pos_full = data[(data[label_columns] == 1).all(1)]
            neg_full = data[(data[label_columns] == -1).all(1)]
            sample_size = min(pos_full.shape[0], neg_full.shape[0])
            pos = pos_full.loc[
                np.random.choice(pos_full.index, sample_size, replace=False)]
            neg = neg_full.loc[
                np.random.choice(neg_full.index, sample_size, replace=False)]
            return pos.append(neg)

    sample_ids = fname.split(".")[-3].split("_")
    #sample_ids.sort(key=lambda x:int(x[1:]))
    with gzip.open(fname) as fp:
        header = fp.readline()
        data = header.split()
        tf_names = [
            x.split("_")[1] for x in header.split() if x.startswith('label')]

    if len(sample_ids) == 1:
        train_samples = sample_ids
        validation_samples = sample_ids
    else:
        num_validation_samples = int(math.ceil(len(sample_ids)/4.0))
        validation_samples = sample_ids[:num_validation_samples]
        train_samples = sample_ids[num_validation_samples:]

    data = pd.read_csv(fname, sep="\s+", index_col=0, compression='gzip')
    label_columns = [x for x in data.columns if x.startswith('label')]
    for label_column in label_columns:
        new_labels = np.zeros(len(data), dtype=int)
        for i, x in enumerate(data[label_column]):
            if isinstance(x, int):
                val = x
            else:
                val = int(x.split(",")[0])
            new_labels[i] = -1 if val <= 0 else 1
        data[label_column] = new_labels

    train_chrs = ['chr%i' %i for i in xrange(5, 23)]
    train = subset_data(data, train_samples, train_chrs)
    validation_chrs = ['chr%i' %i for i in xrange(3, 5)]
    validation = subset_data(data, validation_samples, validation_chrs)
    return balance_data(train), balance_data(validation)

train, validation = load_data(sys.argv[1])
#clf_1 = DecisionTreeClassifier(max_depth=20)
#clf_1 = RandomForestClassifier(max_depth=15)
clf_1 = GradientBoostingClassifier(n_estimators=25)

all_predictors = [ x for x in train.columns
                   if not x.startswith('label') ]
                   #and x != 'access_score' ]
predictors = all_predictors
for label in [x for x in train.columns if x.startswith('label')]:
    print label, predictors
    mo = clf_1.fit(train[predictors], train[label])
    y_hat = mo.predict(train[predictors])
    print label, 'train', (train[label] == y_hat).sum()/float(len(y_hat)), len(y_hat)
    y_hat = mo.predict(validation[predictors])
    positives = np.array(validation[label] == 1)
    negatives = np.array(validation[label] == -1)
    print "pos frac", (y_hat[positives] == 1).sum()/float(positives.sum()), positives.sum()
    print "neg frac", (y_hat[negatives] == -1).sum()/float(negatives.sum()), negatives.sum()
    print 'validation', (validation[label] == y_hat).sum()/float(len(y_hat)), len(y_hat)
    print
