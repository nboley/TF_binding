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
        pos = data[data['label'] == 1]
        neg_full = data[data['label'] == -1]
        neg = neg_full.loc[
            np.random.choice(neg_full.index, pos.shape[0], replace=False)]
        full = pos.append(neg)
        return full
    """
    E114
    E116
    E117
    E118
    """
    data = pd.read_csv(fname, sep="\s+", index_col=0)
    train_cells = ['E116', 'E117', 'E118', 'E123', 'E119', 'E122']
    train_chrs = ['chr%i' %i for i in xrange(5, 23)]
    train = subset_data(data, train_cells, train_chrs)

    test_cells = ['E114', 'E120', 'E121', 'E127']
    test_cells = train_cells
    test_chrs = ['chr%i' %i for i in xrange(3, 5)]
    test = subset_data(data, test_cells, test_chrs)
    return balance_data(train), balance_data(test)

train, test = load_data(sys.argv[1])
clf_1 = DecisionTreeClassifier(max_depth=20)
clf_1 = RandomForestClassifier(max_depth=20)
clf_1 = GradientBoostingClassifier(n_estimators=25)
predictors = [ x for x in train.columns[1:] if x.startswith('CTCF') ]
mo = clf_1.fit(train[predictors], train['label'])

y_hat = mo.predict(train[predictors])
print 'train', (train['label'] == y_hat).sum()/float(len(y_hat)), len(y_hat)

y_hat = mo.predict(test[predictors])
print 'test', (test['label'] == y_hat).sum()/float(len(y_hat)), len(y_hat)
