import sys
sys.setrecursionlimit(50000)

import cPickle as pickle
import numpy as np
import scipy.misc
import os

from pyTFbindtools.sequence import one_hot_encode_sequence
from pyTFbindtools.cross_validation import (
    ClassificationResult, 
    find_optimal_ambiguous_peak_threshold, 
    plot_ambiguous_peaks,
    plot_peak_ranks,
    plot_pr_curve )
from pyTFbindtools.peaks import merge_peaks_and_labels, get_intervals_from_peaks
from ScoreModel import (
    score_convolutions, rank_convolutions,
    get_encode_pwm_hits,
    plot_convolutions )
from get_signal import encode_peaks_sequence_into_array, encode_peaks_bigwig_into_array

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute,
    Flatten )
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.regularizers import l1

import theano.tensor as T
    
def expected_F1_loss(y_true, y_pred, beta=0.1):
    min_label = T.min(y_true)
    max_label = T.max(y_true)
    y_true = (y_true - min_label)/(max_label - min_label)
    
    expected_true_positives = T.sum(y_pred*y_true)
    expected_false_positives = T.sum(y_pred*(1-y_true))
    expected_false_negatives = T.sum((1-y_pred)*y_true)

    precision = expected_true_positives/(
        expected_true_positives + expected_false_positives + 1.0)
    recall = expected_true_positives/(
        expected_true_positives + expected_false_negatives + 1.0)

    return (-1e-6 -(1+beta*beta)*precision*recall)/(beta*beta*precision+recall+2e-6)

def balance_matrices(X, labels, balance_option='downsample'):
    """
    Create equal number of positive and negative examples.

    Parameters
    ----------
    X : ndarray
        Expected shape is (samples, input_dim).
    labels : 1d array
        Expects label of 1 for positives and 0 for negatives.
    balance_option : string, optional
        Method used to balance data.
        Legal values: 'downsample', 'upsample'.
    """
    pos_full = X[(labels == 1)]
    neg_full = X[(labels == 0)]
    if balance_option=='downsample':
        sample_size = min(pos_full.shape[0], neg_full.shape[0])
        pos = pos_full[
            np.random.choice(pos_full.shape[0], sample_size, replace=False)]
        neg = neg_full[
            np.random.choice(neg_full.shape[0], sample_size, replace=False)]
    elif balance_option=='upsample':
        sample_size = max(pos_full.shape[0], neg_full.shape[0])
        pos = pos_full[
            np.random.choice(pos_full.shape[0], sample_size, replace=True)]
        neg = neg_full[
            np.random.choice(neg_full.shape[0], sample_size, replace=True)]
    else:
        raise ValueError('invalid matrix balancing option!')
    return np.vstack((pos, neg)), np.array(
        [1]*sample_size + [0]*sample_size, dtype='float32')

def load_model(fname):
    try:
        with open(fname) as fp:
            return pickle.load(fp)
    except:
        return model_from_json(open(fname, 'r').read())

def set_ambiguous_labels(labels, scores, threshold):
    ambig_labels = (labels == -1)
    labels[ambig_labels] = 0
    labels[ambig_labels&(scores > threshold)] = 1
    return labels

class KerasModelBase():
    def __init__(self, peaks_and_labels, target_metric='recall_at_05_fdr',
                 batch_size=200, num_conv_layers=3, l1_decay=0,
                 num_conv=25, conv_height=5, conv_width=15, dropout=0.2,
                 maxpool_size=35, maxpool_stride=35, gru_size=35, tdd_size=45,
                 model_type='cnn'):
        """
        Base class for Keras model objects.

        Parameters
        ----------
        peaks_and_labels : PeaksAndLabels
        target_metric : string, default: 'recall_at_05_fdr'
            Metric used for model selection.
            Expects an attribute of ClassificationResult.
        batch_size : int, default: 200
        num_conv_layers : int, default: 1
            Number of convolution layers before pooling.
        l1_decay : float, default: 0
            L1 weight decay, applied to all convolutional
            layers except for the first one.
        num_conv : int, default: 25
            Applies same number of convolutions in each layer.
        conv_height : int, default: 4
        conv_width : int, default: 8
        maxpool_size : int, default: 35
            must be integer multiple of maxpool_stride.
        maxpool_stride : int, default: 35
        gru_size: int, default: 35
        tdd_size: int, default: 45
        model_type, string, default: 'cnn'
            Model type. Legal values: 'cnn', 'cnn-rnn'tdd'.
        """
        self.batch_size = batch_size
        self.seq_len = peaks_and_labels.max_peak_width
        self.ambiguous_peak_threshold = None
        self.target_metric = target_metric

        print 'building rSeqDNN architecture...'
        num_conv_outputs = ((self.seq_len - num_conv_layers*conv_width) + num_conv_layers)
        num_maxpool_outputs = int(((num_conv_outputs-maxpool_size)/maxpool_stride)+1)

        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly
        assert maxpool_size%maxpool_stride == 0

        # Define architecture     
        self.model = Sequential()
        self.model.add(Convolution2D(
            num_conv,
            conv_height, conv_width,
            activation="relu", init="he_normal",
            input_shape=(1, conv_height, self.seq_len)
        ))
        self.model.add(Dropout(dropout))
        for i in xrange(1, num_conv_layers):
            self.model.add(Convolution2D(
                num_conv,
                1, conv_width,
                activation="relu", init="he_normal",
                W_regularizer=l1(l1_decay)
                ))
            self.model.add(Dropout(dropout))
        self.model.add(MaxPooling2D(
            pool_size=(1,maxpool_size),
            strides=(1,maxpool_stride)
        ))
        if model_type=='cnn':
            self.model.add(Flatten())
        elif model_type=='cnn-rnn-tdd':
            self.model.add(Reshape((num_conv,num_maxpool_outputs)))
            self.model.add(Permute((2,1)))
            # make the number of max pooling outputs the time dimension
            self.model.add(GRU(output_dim=gru_size,return_sequences=True))
            self.model.add(TimeDistributedDense(tdd_size,activation="relu"))
            self.model.add(Flatten())
        else:
            raise ValueError('invalid model type! supported choices are cnn,cnn-rnn-tdd') 
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    @property
    def curr_model_config_hash(self):
        return abs(hash(str(self.model.get_config())))

    def compile(self, ofname, loss, optimizer, class_mode="binary"):
        loss_name = loss if isinstance(loss, str) else loss.__name__
        fname = "%s.MODEL.%s.json" % (ofname, loss_name)
        print "compiling model..."
        self.model.compile(optimizer,
                           loss,
                           class_mode)
        print("Serializing compiled model." )
        json_string = self.model.to_json()
        open(fname, 'w').write(json_string)

    def predict(self, X_validation, verbose=False):
        preds = self.model.predict_classes(X_validation, verbose=int(verbose))
        # move the predicted labels into 0, 1 space
        preds[preds == 0] = 0
        return preds

    def predict_proba(self, predictors, verbose=False):
        return self.model.predict_proba(predictors, verbose=int(verbose))
    
    def find_optimal_ambiguous_peak_threshold(self, X, y, scores):
        return find_optimal_ambiguous_peak_threshold(
            self, X, y, scores, 20)
    
    def evaluate(self, X_validation, y_validation):
        preds = self.predict(X_validation)
        pred_probs = self.predict_proba(X_validation)
        return ClassificationResult(y_validation, preds, pred_probs)
    
    def evaluate_peaks_and_labels(
            self, 
            data, 
            genome_fasta=None,
            bigwig_fname=None,
            filter_ambiguous_labels=False,
            plot_fname=None):
        '''evaluate model
        '''
        X_validation, y_validation = self.build_predictor_and_label_matrices(
            data, genome_fasta=genome_fasta, bigwig_fname=bigwig_fname,
            add_reverse_complements=False,
            filter_ambiguous_labels=filter_ambiguous_labels)
        # set the ambiguous labels
        if not filter_ambiguous_labels:
            if plot_fname is not None:
                plot_ambiguous_peaks(
                    data.scores[y_validation == 0], 
                    self.predict_proba(X_validation)[y_validation == 0], 
                    plot_fname)
            y_validation = set_ambiguous_labels(
                y_validation, data.scores, self.ambiguous_peak_threshold)
 
        return self.evaluate(X_validation, y_validation)

    def _reshape_coded_seqs_array(self, coded_seqs):
        '''Reshape coded seqs into Keras acceptible format.
        '''
        if len(np.shape(coded_seqs)) == 3:
            return np.reshape(coded_seqs, (len(coded_seqs), 1, 4, self.seq_len))
        else:
            return coded_seqs

    def build_predictor_and_label_matrices(
            self, data, genome_fasta=None, bigwig_fname=None,
            add_reverse_complements=True, filter_ambiguous_labels=True):
        signal_arr_list = []
        if genome_fasta is not None:
            print('loading features from fasta...')
            fasta_array = self._reshape_coded_seqs_array(
                encode_peaks_sequence_into_array(data.peaks, genome_fasta))
            if add_reverse_complements:
                fasta_array = np.concatenate((fasta_array, fasta_array[:, :, ::-1, ::-1]))
            signal_arr_list.append(fasta_array)
        if bigwig_fname is not None:
            print('loading features from bigwig...')
            bigwig_array = encode_peaks_bigwig_into_array(data.peaks, bigwig_fname)
            if add_reverse_complements:
                bigwig_array = np.concatenate((bigwig_array, bigwig_array[:, :, :, ::-1]))
            signal_arr_list.append(bigwig_array)
        X = np.concatenate(signal_arr_list, axis=2)
        y = np.array(data.labels, dtype='float32')
        if add_reverse_complements:
            y = np.concatenate((y, y))
        if filter_ambiguous_labels:
            X = X[y != -1,:,:,:]
            y = y[y != -1]

        return X, y

class KerasModel(KerasModelBase):
    def _fit_with_balanced_data(
            self, X_train, y_train, X_validation, y_validation, ofname):
        b_X_validation, b_y_validation = balance_matrices(
            X_validation, y_validation)
        b_X_train, b_y_train = balance_matrices(X_train, y_train)
        print 'num training positives: ', sum(b_y_train==1)
        print 'num training negatives: ', sum(b_y_train==0)
        
        print("Compiling model with binary cross entropy loss.")
        self.compile(ofname, 'binary_crossentropy', Adam())
        early_stopping = EarlyStopping(monitor='val_loss', patience=6)
        self.model.fit(
                b_X_train, b_y_train,
                validation_data=(b_X_validation, b_y_validation),
                show_accuracy=True,
                class_weight={0:1.0, 1:1.0},
                batch_size=self.batch_size,
                callbacks=[early_stopping])
        print 'Performance on balanced early stopping data:'
        print self.evaluate(b_X_validation, b_y_validation)
        print 'Performance on full unbalanced early stopping data:'
        print self.evaluate(X_validation, y_validation)
        return self

    def _fit(self, X_train, y_train, X_validation, y_validation, numEpochs, ofname):
        neg_class_cnt = (y_train == 0).sum()
        pos_class_cnt = (y_train == 1).sum()
        assert neg_class_cnt + pos_class_cnt == len(y_train)
        class_prbs = dict(zip([0, 1],
                              [len(y_train)/float(neg_class_cnt),
                               len(y_train)/float(pos_class_cnt)]))
        print("Switiching to weighted cross entropy loss function.")
        print("Compiling model with cross entropy loss.")
        weights_ofname = "%s.%s" % (ofname, "fit_weights.hd5")
        self.model.save_weights(weights_ofname, overwrite=True)
        self.compile(ofname, 'binary_crossentropy', Adam())
        self.model.load_weights(weights_ofname)
        res = self.evaluate(X_validation, y_validation)
        self.best_target_metric = getattr(res, self.target_metric)

        for epoch in xrange(numEpochs):
            self.model.fit(
                X_train, y_train,
                validation_data=(X_validation, y_validation),
                show_accuracy=True,
                class_weight=class_prbs,
                batch_size=self.batch_size,
                nb_epoch=1)
            res = self.evaluate(X_validation, y_validation)
            print res

            current_target_metric = getattr(res, self.target_metric)
            if (current_target_metric > self.best_target_metric):
                print("highest %s so far. Saving weights." % self.target_metric)
                self.model.save_weights(weights_ofname, overwrite=True)
                self.best_target_metric = current_target_metric

        # load and return the best model
        print "Loading best model"
        self.model.load_weights(weights_ofname)
        return self

    def train(self, data, ofname, genome_fasta=None,
              bigwig_fname=None, jitter_peaks_by=None, unbalanced_train_epochs=12):
        # split into fitting and early stopping
        data_fitting, data_stopping = next(data.iter_train_validation_subsets())
        if jitter_peaks_by is not None:
            data_fitting_positives = data_fitting.filter_by_label(1)
            jittered_positives = merge_peaks_and_labels(
                data_fitting_positives.jitter_peaks(jitter)
                for jitter in jitter_peaks_by)
            data_fitting = merge_peaks_and_labels((data_fitting, jittered_positives))
        X_validation, y_validation = self.build_predictor_and_label_matrices(
            data_stopping, genome_fasta,
            bigwig_fname, filter_ambiguous_labels=True)

        X_train, y_train = self.build_predictor_and_label_matrices(
            data_fitting, genome_fasta,
            bigwig_fname, filter_ambiguous_labels=True)

        print("Initializing model from balanced training set.")
        self._fit_with_balanced_data(
            X_train, y_train, X_validation, y_validation, ofname)

        print("Fitting full training set with cross entropy loss.")
        self._fit(X_train, y_train, X_validation, y_validation,
                  unbalanced_train_epochs, ofname)

        if any(data.labels==-1):
            # build predictor matrices with  ambiguous labels
            print("Setting the ambiguous labels peak threshold.")
            X_train, y_train = self.build_predictor_and_label_matrices(
                data_fitting, genome_fasta, bigwig_fname, filter_ambiguous_labels=False)
            self.ambiguous_peak_threshold = \
                self.find_optimal_ambiguous_peak_threshold(
                    X_train, y_train, data_fitting.scores)
            y_train = set_ambiguous_labels(
                y_train, data_fitting.scores, self.ambiguous_peak_threshold)

            X_validation, y_validation = self.build_predictor_and_label_matrices(
                data_stopping, genome_fasta, bigwig_fname, filter_ambiguous_labels=False)
            y_validation = set_ambiguous_labels(
                y_validation, data_stopping.scores, self.ambiguous_peak_threshold)
            print self.evaluate(X_validation, y_validation)

            print("Re-fitting the model with the imputed data.")
            self._fit(X_train, y_train, X_validation, y_validation,
                      unbalanced_train_epochs, ofname)

        return self

    def score(self): return self.best_target_metric

    def classification_report(
            self, data, genome_fasta, ofname, filter_ambiguous_labels=True):
        '''plots TP, FP, FN, TN peaks
        '''
        X, y_true = self.build_predictor_and_label_matrices(
            data, genome_fasta, False)
        if filter_ambiguous_labels:
            X = X = X[y_true != -1,:,:,:]
            y_true_scores = data.scores[y_true != -1]
            y_true = y_true[y_true != -1]
        y_pred = self.predict(X).squeeze()
        y_pred_scores = self.predict_proba(X).squeeze()
        plot_peak_ranks(y_pred, y_pred_scores, y_true, y_true_scores, ofname)
        plot_pr_curve(y_true, y_pred_scores, ofname)
        print 'getting encode pwm matches...'
        encode_pwm_hits = get_encode_pwm_hits(self.model)
        encode_hit_names = [[hit[-1] for hit in hits] for hits in encode_pwm_hits]
        print 'ranking convolutions...'
        try:
            convolution_scores = score_convolutions(self.model, X, self.batch_size)
            rank_metrics = ['auROC', 'sensitivity', 'specificity']
            ranks = [rank_convolutions(convolution_scores, y_true, y_pred, metric)
                     for metric in rank_metrics]
            rank_dictionary = dict(zip(rank_metrics, ranks))
        except:
            'convolution ranking skipped!'
            rank_dictionary = None
        print 'plotting convolutions...'
        plot_convolutions(self.model, ofname, rank_dictionary, encode_hit_names)

        return self
