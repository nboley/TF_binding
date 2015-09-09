import os, sys

import itertools

import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
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
        pos.append(neg)
        return pos
    
    data = pd.read_csv(fname, sep="\s+", index_col=0)
    train_cells = ['E116', 'E117', 'E118', 'E123', 'E119', 'E122']
    train_chrs = ['chr%i' %i for i in xrange(5, 23)]
    train = subset_data(data, train_cells, train_chrs)

    test_cells = ['E114', 'E121', 'E120', 'E127']
    test_chrs = ['chr%i' %i for i in xrange(3, 5)]
    test = subset_data(data, test_cells, test_chrs)
    return balance_data(train), balance_data(test)

train, test = load_data(sys.argv[1])
clf_1 = DecisionTreeRegressor(max_depth=20)
predictors = train.columns[1:]
labels = train.columns[1:]
