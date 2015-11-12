### Script to make motif matrix with random forest
### Peyton Greenside
### 11/9/15
#############################################################################################

import os
import sys

import numpy as np
import pandas as pd
import random

import pyTFbindtools.motif_tools
import pysam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from collections import namedtuple
from scipy.stats.mstats import mquantiles
from sklearn.metrics import classification_report

import multiprocessing
import ctypes
from grit.lib.multiprocessing_utils import fork_and_wait

# Get genome
hg19_genome='/mnt/data/annotations/by_organism/human/hg19.GRCh37/hg19.genome.fa'
genome = pysam.FastaFile(hg19_genome)

# Get regions in dictionary
num_feat=100

# region_bed = "/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/atac_seq/merged_matrices/Leuk_35M_counts_per_peak_merged_macsqval5_pseudoreps.txt"
region_bed = "/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/data/peak_lists/peaks_with_any_sig_comparison.txt" # Only peaks with a significant comparison
region_df = pd.read_table(region_bed, header=False)
regions = {}
for r in region_df.index[0:num_feat]:
    regions['{0}:{1}-{2}'.format(region_df.ix[r,0],region_df.ix[r,1]-400,region_df.ix[r,2]+400)]=[region_df.ix[r,0],region_df.ix[r,1]-400,region_df.ix[r,2]+400]

# Get Nathan's motifs
selex_motifs = pyTFbindtools.motif_tools.load_selex_models_from_db()
cisb_motifs = pyTFbindtools.motif_tools.load_pwms_from_db()

# Get one motif for each and choose SELEX
motifs = selex_motifs+[el for el in cisb_motifs if el.tf_id not in [m.tf_id for m in selex_motifs]]

### Implement with grit fork and wait
num_regions = len(regions)
num_motifs = len(motifs)

# Lock
lock_stable = multiprocessing.Lock()

# Counter for regions
rule_index_cntr = multiprocessing.Value('i', 0)

# Store the max scores
max_df = multiprocessing.RawArray(ctypes.c_double,num_regions*num_motifs)
mean_df = multiprocessing.RawArray(ctypes.c_double,num_regions*num_motifs)

def calc_motif_scores_allocate_score(rule_index_cntr, lock_stable, regions, motifs, hg19_genome, max_df, mean_df):
    while True:
        # get the leaf node to work on
        with rule_index_cntr.get_lock():
            rule_index = rule_index_cntr.value
            rule_index_cntr.value += 1
        
        print rule_index
        print len(regions)
        print len(motifs)

        # if this isn't a valid leaf, then we are done
        if rule_index >= len(regions.keys()): 
            return

        key=regions.keys()[rule_index]
        region=regions[key]
        genome = pysam.FastaFile(hg19_genome)
        scores = pyTFbindtools.motif_tools.score_region(region, genome, motifs)
        max_scores = [np.max(s) for s in scores]
        mean_scores = [np.mean(s) for s in scores]

        with lock_stable:

            # Add current rule to training and testing sets
            max_df_iter = np.reshape(max_df, (len(regions), len(motifs)))
            mean_df_iter = np.reshape(mean_df, (len(regions), len(motifs)))
            max_df_iter[rule_index,:]=max_scores
            mean_df_iter[rule_index,:]=mean_scores

            # Update the objects
            max_df[:]=max_df_iter.ravel()
            mean_df[:]=mean_df_iter.ravel()


args = [rule_index_cntr, lock_stable, regions, motifs, hg19_genome, max_df, mean_df]

# Fork worker processes, and wait for them to return
fork_and_wait(10, calc_motif_scores_allocate_score, args)

# get back data frame
max_df = np.reshape(np.array(max_df), (len(regions), len(motifs)))

