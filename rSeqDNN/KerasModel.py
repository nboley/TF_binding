import numpy as np

from pyTFbindtools.sequence import code_seq

from pyTFbindtools.cross_validation import ClassificationResult

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute)
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)

import theano.tensor as T

def expected_F1_loss(y_true, y_pred):
    expected_true_positives = T.sum(y_pred*y_true)
    expected_false_positives = T.sum(y_pred*(1-y_true))
    expected_false_negatives = T.sum((1-y_pred)*y_true)

    precision = expected_true_positives/(
        expected_true_positives + expected_false_positives)
    recall = expected_true_positives/(
        expected_true_positives + expected_false_negatives)

    return -2*precision*recall/(precision + recall)


def encode_peaks_sequence_into_binary_array(peaks, fasta):
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    data = 0.25 * np.ones((len(peaks), 4, pk_width))
    for i, pk in enumerate(peaks):
        seq = fasta.fetch(pk.contig, pk.start, pk.stop)
        # skip sequences overrunning the contig boundary
        if len(seq) != pk_width: continue
        coded_seq = code_seq(seq)
        data[i] = coded_seq[0:4,:]
    return data

def load_model(fname):
    pass

class KerasModelBase():
    def __init__(self, peaks_and_labels):
        self.seq_len = peaks_and_labels.max_peak_width
        numConv = 30
        convStack = 1
        convWidth = 4
        convHeight = 45
        dropoutRate = 0.2
        maxPoolSize = 20
        maxPoolStride = 20
        numConvOutputs = ((self.seq_len - convHeight) + 1)
        numMaxPoolOutputs = int(((numConvOutputs-maxPoolSize)/maxPoolStride)+1)
        gruHiddenVecSize = 35
        numFCNodes = 45
        numOutputNodes = 1
        
        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly
        assert maxPoolSize%maxPoolStride == 0
        
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
        self.model.add(GRU(numConv,gruHiddenVecSize,return_sequences=True))
        self.model.add(TimeDistributedDense(gruHiddenVecSize,numFCNodes))
        self.model.add(Reshape(numFCNodes*numMaxPoolOutputs))
        self.model.add(Dense(numFCNodes*numMaxPoolOutputs,
                             numOutputNodes,
                             activation='sigmoid'))
        optimizer = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        #expected_average_prc
        # binary_crossentropy
        self.model.compile(loss=expected_F1_loss, 
                           optimizer=optimizer,
                           class_mode="binary");

    def _reshape_coded_seqs_array(self, coded_seqs):
        '''Reshape coded seqs into Keras acceptible format.
        '''
        if len(np.shape(coded_seqs)) == 3:
            return np.reshape(coded_seqs, (len(coded_seqs), 1, 4, self.seq_len))
        else:
            return coded_seqs
            
    def evaluate(self, X_validation, y_validation):
        '''evaluate model
        '''
        if len(np.shape(X_validation)) == 3: # reshape to 4D if 3D
            X_validation = self._reshape_coded_seqs_array(X_validation)
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

class KerasModel(KerasModelBase):
    def train_rSeqDNN_model(self,
                            data,
                            genome_fasta,
                            out_filename_prefix,
                            numEpochs=5):
        # split into fitting and early stopping
        data_fitting, data_stopping = next(data.iter_train_validation_subsets())
        X_validation = self._reshape_coded_seqs_array(
                encode_peaks_sequence_into_binary_array(
                        data_stopping.peaks, genome_fasta))
        y_validation = data_stopping.labels
        X_train = self._reshape_coded_seqs_array(
                encode_peaks_sequence_into_binary_array(
                        data_fitting.peaks, genome_fasta))
        y_train = data_fitting.labels
        weights = {len(y_train)/(len(y_train)-y_train.sum()), len(y_train)/y_train.sum()}
        #weights = {1.0, 1.0} # negative, positive
        batch_size = 200
        # fit the model
        best_auPRC = 0
        print("Training...")
        for epoch in xrange(numEpochs):
            self.model.fit(
                X_train, y_train,
                validation_data=(X_validation, y_validation),
                show_accuracy=True,
                class_weight=weights,
                batch_size=batch_size,
                nb_epoch=1)
            res = self.evaluate(X_validation, y_validation)
            print res
            if (res.auPRC > best_auPRC):
                print("highest auPRC accuracy so far. Saving weights.")
                self.model.save_weights(out_filename_prefix,
                                        overwrite=True)
                best_auPRC = res.auPRC

        return self
