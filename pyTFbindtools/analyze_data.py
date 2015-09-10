import os, sys

import itertools

import pandas as pd
import numpy as np
np.random.seed(0)

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
            pos = data[(data[label_columns] == 1).all(1)]
            neg_full = data[(data[label_columns] == -1).all(1)]
            neg = neg_full.loc[
                np.random.choice(neg_full.index, pos.shape[0], replace=False)]
            return pos.append(neg)
    """
    E114
    E116
    E117
    E118
    """
    train_cells = ['E114', 'E123', 'E126']
    test_cells = ['E118']

    data = pd.read_csv(fname, sep="\s+", index_col=0)
    train_chrs = ['chr%i' %i for i in xrange(5, 23)]
    train = subset_data(data, train_cells, train_chrs)

    test_chrs = ['chr%i' %i for i in xrange(3, 5)]
    test = subset_data(data, test_cells, test_chrs)
    return balance_data(train), balance_data(test)

train, test = load_data(sys.argv[1])
clf_1 = DecisionTreeClassifier(max_depth=20)
clf_1 = RandomForestClassifier(max_depth=20)
clf_1 = GradientBoostingClassifier(n_estimators=25)

predictors = [ x for x in train.columns
               if not x.startswith('label') ]
#               and x != 'access_score' ]
for label in [x for x in train.columns if x.startswith('label')]:
    print label
    mo = clf_1.fit(train[predictors], train[label])
    y_hat = mo.predict(train[predictors])
    print label, 'train', (train[label] == y_hat).sum()/float(len(y_hat)), len(y_hat)
    y_hat = mo.predict(test[predictors])
    positives = np.array(test[label] == 1)
    negatives = np.array(test[label] == -1)
    print "pos frac", (y_hat[positives] == 1).sum()/float(positives.sum()), positives.sum()
    print "neg frac", (y_hat[negatives] == -1).sum()/float(negatives.sum()), negatives.sum()
    print 'test', (test[label] == y_hat).sum()/float(len(y_hat)), len(y_hat)
    print
