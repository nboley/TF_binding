import os
import argparse
from collections import namedtuple
from itertools import izip

import numpy as np
from scipy import misc

from pysam import FastaFile

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute)
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)

from pyTFbindtools.peaks import (
    load_summit_centered_peaks, load_narrow_peaks, getFileHandle)
from pyTFbindtools.sequence import code_seq
from pyTFbindtools.cross_validation import (
    iter_train_validation_splits, ClassificationResult)
from pyTFbindtools.DB import load_chipseq_peak_and_matching_DNASE_files_from_db

PeakAndLabel = namedtuple('PeakAndLabel', ['peak', 'sample', 'label'])

def encode_peaks_sequence_into_binary_array(peaks, fasta):
        # find the peak width
        pk_width = peaks[0].pk_width
        # make sure that the peaks are all the same width
        assert all(pk.pk_width == pk_width for pk in peaks)
        data = 0.25 * np.ones((len(peaks), 4, pk_width))
        for i, pk in enumerate(peaks):
            seq = fasta.fetch(pk.contig, pk.start, pk.stop)
            coded_seq = code_seq(seq)
            data[i] = coded_seq[0:4,:]
        
        return data

class PeaksAndLabels():
    def __iter__(self):
        return (
            PeakAndLabel(pk, sample, label) 
            for pk, sample, label 
            in izip(self.peaks, self.samples, self.labels)
        )
    
    @property
    def max_peak_width(self):
        return max(self.peak_widths)
    
    def __init__(self, peaks_and_labels):
        # split the peaks and labels into separate columns. Also
        # keep track of the distinct samples and contigs
        self.peaks = []
        self.samples = []
        self.labels = []
        self.sample_ids = set()
        self.contigs = set()
        self.peak_widths = set()
        for pk, sample, label in peaks_and_labels:
            self.peaks.append(pk)
            self.peak_widths.add(pk.pk_width)
            self.samples.append(sample)
            self.labels.append(label)
            self.sample_ids.add(sample)
            self.contigs.add(pk.contig)
        assert len(self.peak_widths) == 1
        # turn the list of labels into a numpy array
        self.labels = np.array(self.labels, dtype=int)
    
    # One hot encode the sequence in each peak
    def build_coded_seqs(self, fasta):
        return encode_peaks_sequence_into_binary_array(
            self.peaks, fasta)
    
    def subset_data(self, sample_names, contigs):
        '''return data covering sample+names and contigs
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.sample in sample_names
                and pk_and_label.peak.contig in contigs
            )

    def iter_train_validation_subsets(self):
        for train_indices, valid_indices in iter_train_validation_splits(
                self.sample_ids, self.contigs):
            yield (self.subset_data(*train_indices),
                   self.subset_data(*valid_indices))

class KerasModel():
    def __init__(self, peaks_and_labels):
        self.seq_len = peaks_and_labels.max_peak_width
        numConv = 30
        convStack = 1
        convWidth = 4
        convHeight = 45
        dropoutRate = 0.2
        maxPoolSize = 50
        maxPoolStride = 20
        numConvOutputs = ((self.seq_len - convHeight) + 1)
        numMaxPoolOutputs = int(((numConvOutputs-maxPoolSize)/maxPoolStride)+1)
        gruHiddenVecSize = 35
        numFCNodes = 45
        numOutputNodes = 1
        # Define architecture     
        self.model = Sequential()
        self.model.add(Convolution2D(
            numConv, convStack, 
            convWidth, convHeight, activation="relu", init="he_normal"))
        self.model.add(Dropout(dropoutRate))
        self.model.add(MaxPooling2D(poolsize=(1,maxPoolSize),
                                    stride=(1,maxPoolStride)))
        self.model.add(Reshape(numConv,numMaxPoolOutputs))
        self.model.add(Permute((2,1)))
        # make the number of max pooling outputs the time dimension
        self.model.add(GRU(numConv,gruHiddenVecSize,return_sequences=True));
        self.model.add(TimeDistributedDense(gruHiddenVecSize,numFCNodes));
        self.model.add(Reshape(numFCNodes*numMaxPoolOutputs));
        self.model.add(Dense(numFCNodes*numMaxPoolOutputs,numOutputNodes,
                             activation='sigmoid'));
        sgd = SGD(lr=0.01,momentum=0.95,nesterov=True);
        self.model.compile(loss='binary_crossentropy', optimizer=sgd,
                           class_mode="binary");
    
    def get_features(self, coded_seqs):
        '''gets keras-formmated features and labels from data
        '''

        return np.reshape(coded_seqs, (len(coded_seqs), 1, 4, self.seq_len))
        
    def evaluate_rSeqDNN_model(self, X_validation, y_validation):
        '''evaluate model
        '''
        if len(np.shape(X_validation)) == 3: # reshape to 4D if 3D
            X_validation = self.get_features(X_validation)
        preds = self.model.predict_classes(X_validation)
        probs = self.model.predict_proba(X_validation)
        true_pos = y_validation == 1
        true_neg = y_validation == 0
        precision, recall, _ = precision_recall_curve(y_validation, probs)
        prc = np.array([recall,precision])
        auPRC = auc(recall, precision)
        auROC = roc_auc_score(y_validation, probs)
        classification_result = ClassificationResult(
            None, None, None, None, None, None,
            auROC, auPRC, 
            np.sum(preds[true_pos] == 1), np.sum(true_pos),
            np.sum(preds[true_neg] == 0), np.sum(true_neg)
        )

        return classification_result
    

    def train_rSeqDNN_model(self, data, genome_fasta, out_filename_prefix,
                            numEpochs=5):
        # split into fitting and early stopping
        data_fitting, data_stopping = next(data.iter_train_validation_subsets())
        X_validation = self.get_features(data_stopping.build_coded_seqs(genome_fasta))
        y_validation = data_stopping.labels
        X_train = self.get_features(data_fitting.build_coded_seqs(genome_fasta))
        y_train = data_fitting.labels
        weights = {1, 1}
        batch_size = 1500
        # fit the model
        bestBalancedAcc = 0
        print("Training...")
        for epoch in xrange(numEpochs):
            self.model.fit(X_train, y_train,
                      validation_data=(X_validation, y_validation),
                      show_accuracy=True,
                      class_weight=weights,
                      batch_size=batch_size,
                      nb_epoch=1)
            res = self.evaluate_rSeqDNN_model(X_validation, y_validation)
            print res
            if (res.balanced_accuracy > bestBalancedAcc):
                print("highest balanced accuracy so far. Saving weights.")
                self.model.save_weights(out_filename_prefix + ".h5",
                                        overwrite=True)
                bestBalancedAcc = res.balanced_accuracy

        return self


def cross_validated_results(list_of_results):
    ''' 
    '''
    
    return ClassificationResults(list_of_results)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for training rSeqDNN')
    parser.add_argument('--genome-fasta', type=FastaFile, required=True,
                        help='genome file to get sequences')
    parser.add_argument('--pos-regions', type=getFileHandle, required=True,
                        help='regions with positive label')
    parser.add_argument('--neg-regions', type=getFileHandle, required=True,
                        help='regions with negative labels')
    parser.add_argument('--half-peak-width', type=int, default=400,
                        help='half peak width about summits for training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    genome_fasta = args.genome_fasta
    peaks_and_labels = []
    for pos_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.pos_regions), args.half_peak_width):
        peaks_and_labels.append((pos_pk, 'sample', 1))
    for neg_pk in load_summit_centered_peaks(
            load_narrow_peaks(args.neg_regions), args.half_peak_width):
        peaks_and_labels.append((neg_pk, 'sample', 0))
    peaks = PeaksAndLabels(peaks_and_labels)
    model = KerasModel(peaks)
    results = []
    for train, valid in peaks.iter_train_validation_subsets():
        fit_model = model.train_rSeqDNN_model(train, genome_fasta, './test')
        results.append(fit_model.evaluate_rSeqDNN_model(
            valid.build_coded_seqs(genome_fasta), valid.labels))
    print 'Printing cross validation results:'
    print cross_validated_results(results)

if __name__ == '__main__':
    main()
