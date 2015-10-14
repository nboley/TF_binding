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
    roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score)

import theano.tensor as T

def expected_F1_loss(y_true, y_pred):
    expected_true_positives = T.sum(y_pred*y_true)
    expected_false_positives = T.sum(y_pred*(1-y_true))
    expected_false_negatives = T.sum((1-y_pred)*y_true)

    precision = expected_true_positives/(
        expected_true_positives + expected_false_positives + 1.0)
    recall = expected_true_positives/(
        expected_true_positives + expected_false_negatives + 1.0)

    return -2*precision*recall/(precision + recall + 1.0)

def balance_matrices(X, labels):
    pos_full = X[(labels == 1)]
    neg_full = X[(labels == 0)]
    sample_size = min(pos_full.shape[0], neg_full.shape[0])
    pos = pos_full[
        np.random.choice(pos_full.shape[0], sample_size, replace=False)]
    neg = neg_full[
        np.random.choice(neg_full.shape[0], sample_size, replace=False)]
    return np.vstack((pos, neg)), np.array([1]*sample_size + [0]*sample_size)

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
            convWidth, convHeight, 
            activation="linear", init="he_normal"))
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

    def compile(self, loss, optimizer, class_mode="binary"):
        print("Conpiling model (%s, %s, %s)" % (loss, optimizer, class_mode))
        self.model.compile(loss=loss, 
                           optimizer=optimizer,
                           class_mode=class_mode);

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
        f1 = f1_score(y_validation, preds)
        classification_result = ClassificationResult(
            None, None, None, None, None, None,
            auROC, auPRC, f1,
            np.sum(preds[true_pos] == 1), np.sum(true_pos),
            np.sum(preds[true_neg] == 0), np.sum(true_neg)
        )

        return classification_result

class KerasModel(KerasModelBase):
    def train_rSeqDNN_model(self,
                            data,
                            genome_fasta,
                            out_filename,
                            numEpochs=5):
        batch_size = 200

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
        weights = {len(y_train)/(len(y_train)-y_train.sum()), 
                   len(y_train)/y_train.sum()}

        b_X_validation, b_y_validation = balance_matrices(
            X_validation, y_validation)
        b_X_train, b_y_train = balance_matrices(X_train, y_train)

        
        print("Initializing model from balanced training set.")
        self.compile('binary_crossentropy', Adam())
        self.model.fit(
                b_X_train, b_y_train,
                validation_data=(b_X_validation, b_y_validation),
                show_accuracy=True,
                class_weight={1.0, 1.0},
                batch_size=batch_size,
                nb_epoch=3)
        print self.evaluate(b_X_validation, b_y_validation)
        
        print("Fitting the model with unbalanced training set.")
        best_F1 = 0
        self.compile(expected_F1_loss, Adam()) # expected_F1_loss
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
            if (res.F1 > best_F1):
                print("highest auPRC accuracy so far. Saving weights.")
                self.model.save_weights(out_filename,
                                        overwrite=True)
                best_F1 = res.F1

        # load the best model
        print "Loading best model"
        self.model.load_weights(out_filename)
        
        return self