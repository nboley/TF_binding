import sys
sys.setrecursionlimit(50000)

import cPickle as pickle
import numpy as np
import json

from pyTFbindtools.sequence import code_seq

from pyTFbindtools.cross_validation import (
    ClassificationResult, 
    find_optimal_ambiguous_peak_threshold, 
    plot_ambiguous_peaks )

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute)
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_yaml

import theano.tensor as T

def expected_F1_loss(y_true, y_pred):
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

    return (-1e-6 -2*precision*recall)/(precision + recall + 2e-6)

def balance_matrices(X, labels):
    pos_full = X[(labels == 1)]
    neg_full = X[(labels < 1)]
    sample_size = min(pos_full.shape[0], neg_full.shape[0])
    pos = pos_full[
        np.random.choice(pos_full.shape[0], sample_size, replace=False)]
    neg = neg_full[
        np.random.choice(neg_full.shape[0], sample_size, replace=False)]
    return np.vstack((pos, neg)), np.array(
        [1]*sample_size + [-1]*sample_size, dtype='float32')

def encode_peaks_sequence_into_binary_array(peaks, fasta):
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    data = 0.25 * np.ones((len(peaks), 4, pk_width), dtype='float32')
    for i, pk in enumerate(peaks):
        seq = fasta.fetch(pk.contig, pk.start, pk.stop)
        # skip sequences overrunning the contig boundary
        if len(seq) != pk_width: continue
        coded_seq = code_seq(seq)
        data[i] = coded_seq[0:4,:]
    return data

def load_model(fname):
    pass

def set_ambiguous_labels(labels, scores, threshold):
    ambig_labels = (labels == 0)
    labels[ambig_labels] = -1
    labels[ambig_labels&(scores > threshold)] = 1
    return labels

class KerasModelBase():
    def __init__(self, peaks_and_labels, model_def_file=None, batch_size=200):

        self.batch_size = batch_size
        self.seq_len = peaks_and_labels.max_peak_width

        if (model_def_file is not None):
            self.model_def_file = model_def_file
            self.parse_model_def_file() # parse and update self.model
        else: # resort to defaults
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
                numConv, 
                convWidth, convHeight, 
                activation="relu", init="he_normal", 
                input_shape=(1, 4, self.seq_len)
            ))
            self.model.add(MaxPooling2D(
                pool_size=(1,maxPoolSize), 
                stride=(1,maxPoolStride)
            ))
            self.model.add(Reshape((numConv,numMaxPoolOutputs)))
            self.model.add(Permute((2,1)))
            # make the number of max pooling outputs the time dimension
            self.model.add(GRU(output_dim=gruHiddenVecSize,return_sequences=True))
            self.model.add(TimeDistributedDense(numFCNodes,activation="relu"))
            self.model.add(Reshape((numFCNodes*numMaxPoolOutputs,)))
            self.model.add(Dense(numOutputNodes,activation='sigmoid'))

    @property
    def curr_model_config_hash(self):
        return abs(hash(str(self.model.get_config())))

    def parse_model_def_file(self):
        model_dict = json.loads(open(self.model_def_file,"rb").read())
        # parse architecture parameters
        architecture = model_dict["architecture"]
        num_conv = architecture["num-conv"];
        conv_stack = architecture["conv-stack"];
        conv_height = architecture["conv-height"];
        conv_width = architecture["conv-width"];
        dropout_rate = architecture["dropout"];
        max_pool_size = architecture["maxpool-size"];
        max_pool_stride = architecture["maxpool-stride"];
        num_conv_out = ((self.seq_len - conv_height) + 1);
        num_max_pool_out = int(((num_conv_out-max_pool_size)/max_pool_stride)+1)
        # if CNN-RNN is specified, parse GRU size
        if (architecture["type"] == "cnn-rnn"): 
            gru_hidden_vec_size = architecture["gru-hidden-size"];
            num_fc_nodes = architecture["num-fc-nodes"];

        # setup Sequential() model
        self.model = Sequential()
        self.model.add(Convolution2D(
            num_conv, conv_width, conv_height, 
            activation="relu", init="he_normal", 
            input_shape=(conv_stack, 4, self.seq_len)
        ))
        self.model.add(MaxPooling2D(
            pool_size=(1,max_pool_size), 
            stride=(1,max_pool_stride)
        ))
        if (architecture["type"] == "cnn-rnn"):
            self.model.add(Reshape((num_conv,num_max_pool_out)))
            self.model.add(Permute((2,1)))
            # make the number of max pooling outputs the time dimension
            self.model.add(GRU(output_dim=gru_hidden_vec_size,return_sequences=True))
            self.model.add(TimeDistributedDense(num_fc_nodes,activation="relu"))
            self.model.add(Reshape((num_fc_nodes*num_max_pool_out,)))
            self.model.add(Dense(1,activation='sigmoid'))
        else:
            self.model.add(Reshape((num_conv*num_max_pool_out,)))
            self.model.add(Dense(1,activation='sigmoid'))
    
    def compile(self, loss, optimizer, use_model_file, class_mode="binary"):
        loss_name = loss if isinstance(loss, str) else loss.__name__
        fname = "MODEL.%s.%s.obj" % (self.curr_model_config_hash, loss_name)
        if use_model_file:
            try:
                print "Loading pickled model '%s'" % fname 
                with open(fname) as fp:
                    self.model = pickle.load(fp)
                print "Finished loading pickled model."
            except IOError:
                raise ValueError("ERROR loading picked model!exiting!")
        else:
            print "compiling model..."
            self.model.compile(loss=loss, 
                               optimizer=optimizer,
                               class_mode=class_mode)
        if use_model_file:    
            print("Saving compiled model to pickle object." )
            with open(fname, "w") as fp:
                pickle.dump(self.model, fp)

    def predict(self, X_validation, verbose=False):
        preds = self.model.predict_classes(X_validation, verbose=int(verbose))
        # move the predicted labels into -1, 1 space
        preds[preds == 0] = -1
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
            genome_fasta, 
            filter_ambiguous_labels=False,
            plot_fname=None):
        '''evaluate model
        '''
        X_validation, y_validation = self.build_predictor_and_label_matrices(
            data, genome_fasta, filter_ambiguous_labels=filter_ambiguous_labels)
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
            self, data, genome_fasta, filter_ambiguous_labels):
        X = self._reshape_coded_seqs_array(
                encode_peaks_sequence_into_binary_array(
                    data.peaks, genome_fasta))
        y = np.array(data.labels, dtype='float32')
        if filter_ambiguous_labels:
            X = X[y != 0,:,:,:]
            y = y[y != 0]

        return X, y


class KerasModel(KerasModelBase):
    def _fit_with_balanced_data(
            self, X_train, y_train, X_validation, y_validation, numEpochs,
            use_model_file):
        b_X_validation, b_y_validation = balance_matrices(
            X_validation, y_validation)
        b_X_train, b_y_train = balance_matrices(X_train, y_train)        
        
        print("Compiling model with binary cross entropy loss.")
        self.compile('binary_crossentropy', Adam(), use_model_file)
        self.model.fit(
                b_X_train, b_y_train,
                validation_data=(b_X_validation, b_y_validation),
                show_accuracy=True,
                class_weight={1.0, 1.0},
                batch_size=self.batch_size,
                nb_epoch=numEpochs)
        print self.evaluate(b_X_validation, b_y_validation)
        return self

    def _fit(self, X_train, y_train, X_validation, y_validation, numEpochs,
             use_model_file):
        neg_class_cnt = (y_train == -1).sum()
        pos_class_cnt = (y_train == 1).sum()
        assert neg_class_cnt + pos_class_cnt == len(y_train)
        class_prbs = {float(neg_class_cnt)/len(y_train), 
                      float(pos_class_cnt)/len(y_train)}

        print("Switiching to F1 loss function.")
        print("Compiling model with expected F1 loss.")
        self.compile(expected_F1_loss, Adam(), use_model_file)
        best_auPRC = 0
        best_F1 = 0
        best_average = 0

        if self.model_def_file:
            out_filename = (self.model_def_file).split(".json")[0]
            out_filename = out_filename + ".obj"
        else:
            out_filename = "fit_weights.obj"

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

            if (res.F1 + res.auPRC > best_average):
                print("highest auPRC + F1 score so far. Saving weights.")
                self.model.save_weights(out_filename, overwrite=True)
                best_average = res.F1 + res.auPRC
            if (res.auPRC > best_auPRC):
                best_auPRC = res.auPRC
            if (res.F1 > best_F1):
                best_F1 = res.F1                

        # load and return the best model
        print "Loading best model"
        self.model.load_weights(out_filename)
        return self

    def train(self, data, genome_fasta, out_filename, use_model_file,
              balanced_train_epochs=3, unbalanced_train_epochs=8):
        # split into fitting and early stopping
        data_fitting, data_stopping = next(data.iter_train_validation_subsets())
        X_validation, y_validation = self.build_predictor_and_label_matrices(
            data_stopping, genome_fasta, filter_ambiguous_labels=True)

        X_train, y_train = self.build_predictor_and_label_matrices(
            data_fitting, genome_fasta, filter_ambiguous_labels=True)

        print("Initializing model from balanced training set.")
        self._fit_with_balanced_data(
            X_train, y_train, X_validation, y_validation,
            balanced_train_epochs, use_model_file)
        
        print("Fitting full training set with expected F1 loss.")
        self._fit(X_train, y_train, X_validation, y_validation,
                  unbalanced_train_epochs, use_model_file)
        
        # build the predictor matrixes, including the ambiguous labels
        print("Setting the ambiguous labels peak threshold.")
        X_train, y_train = self.build_predictor_and_label_matrices(
            data_fitting, genome_fasta, filter_ambiguous_labels=False)
        self.ambiguous_peak_threshold = \
            self.find_optimal_ambiguous_peak_threshold(
                X_train, y_train, data_fitting.scores)
        y_train = set_ambiguous_labels(
            y_train, data_fitting.scores, self.ambiguous_peak_threshold)

        X_validation, y_validation = self.build_predictor_and_label_matrices(
            data_stopping, genome_fasta, filter_ambiguous_labels=False)
        y_validation = set_ambiguous_labels(
            y_validation, data_stopping.scores, self.ambiguous_peak_threshold)
        print self.evaluate(X_validation, y_validation)

        print("Re-fitting the model with the imputed data.")
        self._fit(X_train, y_train, X_validation, y_validation,
                  unbalanced_train_epochs, use_model_file)
        
        return self

