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
from ScoreModel import (
    score_convolutions, rank_convolutions,
    get_encode_pwm_hits,
    plot_convolutions )
from get_signal import encode_peaks_sequence_into_array, encode_peaks_bigwig_into_array

from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import model_from_json, Sequential, Graph
from keras.layers.core import (
    Dense, Dropout, Activation, Reshape,TimeDistributedDense, Permute,
    Flatten, Merge )
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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
    X : ndarray or sequence of ndarray
        Expected shape is (samples, 1, height, length),
        uniform samples and length in ndarrays.
    labels : 1d array
        Expects label of 1 for positives and 0 for negatives.
    balance_option : string, optional
        Method used to balance data.
        Legal values: 'downsample', 'upsample'.
    Returns
    -------
    X_balanced : ndarray or sequence of ndarray
        ndarrays balanced uniformly if a sequence.
    y : 1d array
        balanced labels.
    """
    if isinstance(X, list):
        heights = [np.shape(x)[2] for x in X]
        X = np.concatenate(X, axis=2)
    pos_indxs = np.where(labels==1)[0]
    neg_indxs = np.where(labels==0)[0]
    pos_size = len(pos_indxs)
    neg_size = len(neg_indxs)
    if balance_option=='downsample':
        sample_size = min(pos_size, neg_size)
        if neg_size > pos_size:
            pos = np.take(X, pos_indxs, axis=0)
            neg = np.take(X, np.random.choice(neg_indxs, pos_size, replace=False), axis=0)
        else:
            neg = np.take(X, neg_indxs, axis=0)
            pos = np.take(X, np.random.choice(pos_indxs, neg_size, replace=False), axis=0)
    elif balance_option=='upsample':
        sample_size = max(pos_full.shape[0], neg_full.shape[0])
        pos = pos_full[
            np.random.choice(pos_full.shape[0], sample_size, replace=True)]
        neg = neg_full[
            np.random.choice(neg_full.shape[0], sample_size, replace=True)]
    else:
        raise ValueError('invalid matrix balancing option!')
    y = np.array([1]*sample_size + [0]*sample_size, dtype='float32')
    if isinstance(X, list):
        X_arr_balanced = np.vstack((pos, neg))
        cum_height = np.cumsum([0]+heights)
        X_balanced = []
        for i in range(len(cum_height)-1):
            X_balanced.append(X_arr_balanced[:,:,cum_height[i]:cum_height[i+1],:])
        return X_balanced, y
    else:
        X_balanced = np.vstack((pos, neg))
        return X_balanced, y

def load_model(fname):
    '''loads model saved as json or pickle file.
    '''
    if 'json' in fname:
        return model_from_json(open(fname).read())
    else:
        with open(fname, 'r') as fp:
            return pickle.load(fp)

def set_ambiguous_labels(labels, scores, threshold):
    ambig_labels = (labels == -1)
    labels[ambig_labels] = 0
    labels[ambig_labels&(scores > threshold)] = 1
    return labels

class KerasModelBase():
    def __init__(self, arrays_shapes=None, model_fname=None,
                 multi_mode=False, target_metric='auPRC',
                 batch_size=200, num_conv_layers=3, l1_decay=0,
                 num_conv=15, conv_height=5, conv_width=15, dropout=0.0,
                 maxpool_size=35, maxpool_stride=35, gru_size=35, tdd_size=45,
                 model_type='cnn'):
        """
        Base class for Keras model objects.
        Note: either arrays_shapes or model_fname have to be provided.

        Parameters
        ----------
        arrays_shapes : list, optional
            list of shape tuples for each input array modality.
        model_fname : keras model, optional
            can be initialized from keras architecture.
        stack_arrays : boolean, default: true
            if true, input arrays are stacked into multi-channel format.
            if false, each array connected to separate convolutions.
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
        assert model_fname is not None or arrays_shapes is not None, \
            "Either model_fname or arrays_shapes needed to initialize KerasModel!"
        self.stack_arrays = not multi_mode
        self.batch_size = batch_size
        self.ambiguous_peak_threshold = None
        self.target_metric = target_metric
        # load model if provided, otherwise build model
        if model_fname is not None:
            self.model = load_model(model_fname)
            return
        # check uniform samples and length dimensions
        assert all(shape[0] == arrays_shapes[0][0] for shape in arrays_shapes)
        assert all(shape[-1] == arrays_shapes[0][-1] for shape in arrays_shapes)
        length = arrays_shapes[0][-1]

        print 'building rSeqDNN architecture...'
        num_conv_outputs = ((length - num_conv_layers*conv_width) + num_conv_layers)
        num_maxpool_outputs = int(((num_conv_outputs-maxpool_size)/maxpool_stride)+1)

        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly
        assert maxpool_size%maxpool_stride == 0

        # Define architecture
        self.model = Sequential()
        if self.stack_arrays:
            stacked_height = sum(shape[-2] for shape in arrays_shapes)
            self.model.add(Convolution2D(
                num_conv,
                stacked_height, conv_width,
                activation="relu", init="he_normal",
                input_shape=(1, stacked_height, length)
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
        else:
            unimodal_cnns = []
            # build cnn for each mode
            for i, shape in enumerate(arrays_shapes):
                cnn = Sequential()
                height = shape[-2]
                cnn.add(Convolution2D(
                    num_conv,
                    height, conv_width,
                    activation="relu", init="he_normal",
                    input_shape=shape[1:]))
                cnn.add(Dropout(dropout))
                for i in xrange(1, num_conv_layers-1):
                    cnn.add(Convolution2D(
                        num_conv,
                        1, conv_width,
                        activation="relu", init="he_normal",
                        W_regularizer=l1(l1_decay)))
                    cnn.add(Dropout(dropout))
                unimodal_cnns.append(cnn)
            # concatenate unimodal cnns
            self.model.add(Merge(unimodal_cnns, mode='concat', concat_axis=2))
            # run conv layer on combined output
            self.model.add(Convolution2D(
                num_conv,
                len(arrays_shapes), conv_width,
                activation="relu", init="he_normal",
                W_regularizer=l1(l1_decay)))
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
        print "compiling model..."
        self.model.compile(optimizer,
                           loss,
                           class_mode)
        print("Serializing compiled model." )
        loss_name = loss if isinstance(loss, str) else loss.__name__
        if not isinstance(self.model.layers[0], Merge):
            fname = "%s.MODEL.%s.json" % (ofname, loss_name)
            json_string = self.model.to_json()
            open(fname, 'w').write(json_string)
        else:
        # json is not stable for some merged models
        # pickle instead
            fname = "%s.MODEL.%s.pkl" % (ofname, loss_name)
            with open(fname, 'w') as fp:
                pickle.dump(self.model, fp)

    def predict(self, X_validation, verbose=False):
        if self.stack_arrays and isinstance(X_validation, list):
            return self.model.predict_classes(np.concatenate(X_validation, axis=2),
                                              verbose=int(verbose))
        else:
            return self.model.predict_classes(X_validation, verbose=int(verbose))

    def predict_proba(self, X_validation, verbose=False):
        if self.stack_arrays and isinstance(X_validation, list):
            return self.model.predict_proba(np.concatenate(X_validation, axis=2),
                                            verbose=int(verbose))
        else:
            return self.model.predict_proba(X_validation, verbose=int(verbose))

    def find_optimal_ambiguous_peak_threshold(self, X, y, scores):
        return find_optimal_ambiguous_peak_threshold(
            self, X, y, scores, 20, self.target_metric)

    def evaluate(self, X_validation, y_validation):
        preds = self.predict(X_validation)
        pred_probs = self.predict_proba(X_validation)
        return ClassificationResult(y_validation, preds, pred_probs)

    def evaluate_peaks_and_labels(self, X, labels, include_ambiguous_labels=False,
                                  scores=None, plot_fname=None):
        """
        Evaluate model performance.

        Parameters
        ----------
        X : sequence of ndarray
        labels : 1d array
        include_ambiguous_labels : boolean, default: False
        scores : 1d array, required if include_ambiguous_labels
        plot_fname : str, optional

        Returns
        -------
        ClassificationResult, optionally plots ambiguous examples.
        """
        y_true = np.copy(labels)
        if include_ambiguous_labels:
            assert scores is not None, \
                "model evaluation with ambiguous labels: must include scores!"
            if plot_fname is not None:
                plot_ambiguous_peaks(
                    scores[y_true == -1],
                    self.predict_proba(X)[y_true == -1],
                    plot_fname)
            y_true = set_ambiguous_labels(
                y_true, scores, self.ambiguous_peak_threshold)
        else:
            if np.sum(y_true==-1) > 0:
                X = [X_arr[y_true != -1,:,:,:] for X_arr in X]
                y_true = y_true[y_true != -1]

        return self.evaluate(X, y_true)

class KerasModel(KerasModelBase):
    def _fit_with_balanced_data(
            self, X_train, y_train, X_validation, y_validation, ofname):
        b_X_validation, b_y_validation = balance_matrices(X_validation, y_validation)
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
        epochs_no_imporvement = 0

        while epochs_no_imporvement < numEpochs:
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
                epochs_no_imporvement = 0
            else:
                epochs_no_imporvement += 1

        # load and return the best model
        print "Loading best model"
        self.model.load_weights(weights_ofname)
        return self

    def train(self, fitting_arrays, fitting_labels, fitting_scores,
              stopping_arrays, stopping_labels, stopping_scores, ofname,
              unbalanced_train_epochs=3):
        # filter ambiguous examples
        y_train = np.copy(fitting_labels)
        y_validation = np.copy(stopping_labels)
        if self.stack_arrays:
            fitting_arrays = [np.concatenate(fitting_arrays, axis=2)]
            stopping_arrays = [np.concatenate(stopping_arrays, axis=2)]
        if any(fitting_labels==-1) or any(stopping_labels==-1):
            X_train = [X[fitting_labels != -1,:,:,:] for X in fitting_arrays]
            y_train = y_train[y_train != -1]
            X_validation = [X[stopping_labels != -1,:,:,:] for X in stopping_arrays]
            y_validation = y_validation[y_validation != -1]
        else:
            X_train = fitting_arrays
            X_validation = stopping_arrays
        # fit calls with sequence of input arrays
        #print("Initializing model from balanced training set.")
        #self._fit_with_balanced_data(
        #    X_train, y_train, X_validation, y_validation, ofname)

        print("Fitting full training set with cross entropy loss.")
        self._fit(X_train, y_train, X_validation, y_validation,
                  unbalanced_train_epochs, ofname)

        if any(fitting_labels==-1) or any(stopping_labels==-1):
            print("Starting training with ambiguous examples...")
            print("Setting the ambiguous labels peak threshold...")
            X_train = fitting_arrays
            y_train = np.copy(fitting_labels)
            X_validation = stopping_arrays
            y_validation = np.copy(stopping_labels)
            self.ambiguous_peak_threshold = \
                self.find_optimal_ambiguous_peak_threshold(
                    X_train, y_train, fitting_scores)
            y_train = set_ambiguous_labels(
                y_train, fitting_scores, self.ambiguous_peak_threshold)

            y_validation = set_ambiguous_labels(
                y_validation, stopping_scores, self.ambiguous_peak_threshold)
            print self.evaluate(X_validation, y_validation)

            print("Re-fitting the model with the imputed ambiguous examples...")
            self._fit(X_train, y_train, X_validation, y_validation,
                      unbalanced_train_epochs, ofname)

        return self

    def score(self): return self.best_target_metric

    def classification_report(self, X, labels, ofname, scores=None):
        """
        Generate plots summarizing model behavior.

        Parameters
        ----------
        X : sequence of ndarrays
            model inputs.
        labels : 1d array
        ofname : string
        scores : 1d array, optional

        Returns
        -------
        Precision recall curve,
        ranked sequence filter plots matched to known motifs
        """
        y_true = np.copy(labels)
        if any(labels == -1):
            if scores is not None:
                y_true = set_ambiguous_labels(
                    y_true, scores, self.ambiguous_peak_threshold)
            else:
                X = [X_arr[y_true != -1,:,:,:] for X_arr in X]
                y_true = y_true[y_true != -1]
        y_pred = self.predict(X).squeeze()
        y_pred_scores = self.predict_proba(X).squeeze()
        if scores is not None:
            plot_peak_ranks(y_pred, y_pred_scores, y_true, scores, ofname)
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

def class_weights(y):
    """
    num_samples / class_frequency
    """
    assert len(np.shape(y))==1
    neg_count = float((y==0).sum())
    pos_count = float((y==1).sum())
    neg_weight = len(y)/neg_count
    pos_weight = len(y)/pos_count
    return neg_weight, pos_weight

def sample_weights(y):
    """
    0 for -1 (masked) labels, 1 otherwise
    """
    assert len(np.shape(y))==1
    masked_y = y!=-1
    return masked_y.astype(float)

class KerasModelMultitask():
    def __init__(self, num_tasks, arrays_shapes=None, model_fname=None,
                 multi_mode=False, target_metric='auPRC',
                 batch_size=200, num_conv_layers=3, l1_decay=0,
                 num_conv=15, conv_height=5, conv_width=15, dropout=0.0,
                 maxpool_size=35, maxpool_stride=35, gru_size=35, tdd_size=45,
                 model_type='cnn'):
        """
        Base class for Keras model objects.
        Note: either arrays_shapes or model_fname have to be provided.
        Parameters
        ----------
        num_tasks: int
            number of tasks or outputs.
        arrays_shapes : list, optional
            list of shape tuples for each input array modality.
        model_fname : keras model, optional
            can be initialized from keras architecture.
        stack_arrays : boolean, default: true
            if true, input arrays are stacked into multi-channel format.
            if false, each array connected to separate convolutions.
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
        assert model_fname is not None or arrays_shapes is not None, \
            "Either model_fname or arrays_shapes needed to initialize KerasModel!"
        self.num_tasks = num_tasks
        self.stack_arrays = not multi_mode
        self.batch_size = batch_size
        self.ambiguous_peak_threshold = None
        self.target_metric = target_metric
        # load model if provided, otherwise build model
        if model_fname is not None:
            self.model = load_model(model_fname)
            return
        # check uniform samples and length dimensions
        assert all(shape[0] == arrays_shapes[0][0] for shape in arrays_shapes)
        assert all(shape[-1] == arrays_shapes[0][-1] for shape in arrays_shapes)
        length = arrays_shapes[0][-1]

        print 'building multitask architecture...'
        num_conv_outputs = ((length - num_conv_layers*conv_width) + num_conv_layers)
        num_maxpool_outputs = int(((num_conv_outputs-maxpool_size)/maxpool_stride)+1)

        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly
        assert maxpool_size%maxpool_stride == 0

        # Define architecture
        self.model = Graph()
        if self.stack_arrays:
            stacked_height = sum(shape[-2] for shape in arrays_shapes)
            self.model.add_input(name="input", input_shape=(1, stacked_height, length))
            self.model.add_node(Convolution2D(
                num_conv,
                stacked_height, conv_width,
                activation="relu", init="he_normal",
                #input_shape=(1, stacked_height, length)
            ), name="conv0", input="input")
            self.model.add_node(Dropout(dropout), name="drop0", input="conv0")
            for i in xrange(1, num_conv_layers):
                self.model.add_node(Convolution2D(
                    num_conv,
                    1, conv_width,
                    activation="relu", init="he_normal",
                    W_regularizer=l1(l1_decay)
                ), name="conv%s" % (str(i)), input="drop%s" % (str(i-1)))
                self.model.add_node(Dropout(dropout), name="drop%s" % (str(i)), input="conv%s" % (str(i)))
        else: #TODO: write multimoding for Graph model
            raise RuntimeError("Multimoding is not implemented for multitask models!")
        self.model.add_node(MaxPooling2D(
                pool_size=(1,maxpool_size),
                strides=(1,maxpool_stride)
            ), name="maxpool", input="conv%s" % (str(num_conv_layers-1)))
        if model_type=='cnn':
            self.model.add_node(Flatten(), name="flatten", input="maxpool")
        elif model_type=='cnn-rnn-tdd':
            raise RuntimeError("cnn-rnn-tdd is implemented for multitask models!")
        else:
            raise ValueError('invalid model type! supported choices are cnn,cnn-rnn-tdd')
        for i in xrange(self.num_tasks):
            self.model.add_node(Dense(1), name="dense%s" % (str(i)), input="flatten")
            self.model.add_node(Activation('sigmoid'), name="sigmoid%s" % (str(i)), input="dense%s" % (str(i)))
            self.model.add_output(name="output%s" % (str(i)), input="sigmoid%s" % (str(i)))

    def compile(self, ofname, loss, optimizer):
        print "compiling model..."
        self.model.compile(optimizer, loss)
        print("Serializing compiled model." )
        loss_name = loss["output0"] if isinstance(loss["output0"], str) else loss["output0"].__name__
        fname = "%s.MODEL.%s.json" % (ofname, loss_name)
        json_string = self.model.to_json()
        open(fname, 'w').write(json_string)

    def evaluate(self, validation_data_dict):
        pred_probs = self.model.predict(validation_data_dict)
        preds = dict(zip([name for name in self.model.output_order],
                         [(pred_probs[name]>0.5).astype(float) for name in self.model.output_order]))
        sample_masks = dict(zip([name for name in self.model.output_order],
                                [sample_weights(validation_data_dict[name]).astype(bool)
                                 for name in self.model.output_order]))
        return [ClassificationResult(validation_data_dict[name][sample_masks[name]],
                                     preds[name][sample_masks[name]],
                                     pred_probs[name][sample_masks[name]])
                for name in self.model.output_order]

    def reshape_input_arrays(self, arrays):
        """
        convert list of arrays to proper input arrays.
        """
        if isinstance(arrays, list) and len(arrays)==1:
            return arrays[0]
        elif self.stack_arrays:
            return np.concatenate(fitting_arrays, axis=2)
        else:
            return arrays

    def evaluate_peaks_and_labels(self, X, labels, include_ambiguous_labels=False,
                                  scores=None, plot_fname=None):
        """
        Creates data dictionary and calls evaluate.
        TODO
        this is as somewhat unnecessary method with unnecessary kwargs
        refactor code
        """
        assert self.num_tasks==np.shape(labels)[-1]
        X = self.reshape_input_arrays(X)
        data_dict = dict(zip(["input"]+["output%s" % (i) for i in xrange(self.num_tasks)],
                             [X]+[labels[:, i] for i in xrange(self.num_tasks)]))
        return self.evaluate(data_dict)


    def train(self, fitting_arrays, fitting_labels, fitting_scores,
              stopping_arrays, stopping_labels, stopping_scores, ofname,
              unbalanced_train_epochs=3):
        assert self.num_tasks==np.shape(stopping_labels)[-1]
        # define X and y arrays
        y_train = fitting_labels
        y_validation = stopping_labels
        X_train = self.reshape_input_arrays(fitting_arrays)
        X_validation = self.reshape_input_arrays(stopping_arrays)
        # define input_output dictionary
        train_data_dict = dict(zip(["input"]+["output%s" % (i) for i in xrange(self.num_tasks)],
                                   [X_train]+[y_train[:, i] for i in xrange(self.num_tasks)]))
        validation_data_dict = dict(zip(["input"]+["output%s" % (i) for i in xrange(self.num_tasks)],
                                        [X_validation]+[y_validation[:, i] for i in xrange(self.num_tasks)]))
        # compute class and sample weights
        class_weight_dict = dict(zip(["output%s" % (i) for i in xrange(self.num_tasks)],
                                     [dict(zip([0, 1], class_weights(y_train[:, i])))
                                      for i in xrange(self.num_tasks)]))
        sample_weight_dict = dict(zip(["output%s" % (i) for i in xrange(self.num_tasks)],
                                      [sample_weights(y_train[:, i]) for i in xrange(self.num_tasks)]))
        # compile and fit
        weights_ofname = "%s.%s" % (ofname, "fit_weights.hd5")
        self.model.save_weights(weights_ofname, overwrite=True)

        print("Compiling model with cross entropy loss.")
        loss_dict = dict(zip(["output%s" % (i) for i in xrange(self.num_tasks)],
                             ["binary_crossentropy" for i in xrange(self.num_tasks)]))
        self.compile(ofname, loss_dict, Adam())
        self.model.load_weights(weights_ofname)
        res_list = self.evaluate(validation_data_dict)
        self.best_target_metric = sum(getattr(res, self.target_metric) for res in res_list)
        epochs_no_imporvement = 0

        while epochs_no_imporvement < unbalanced_train_epochs:
            self.model.fit(
                train_data_dict,validation_data=validation_data_dict,
                class_weight=class_weight_dict, sample_weight=sample_weight_dict,
                batch_size=self.batch_size, nb_epoch=1)
            res_list = self.evaluate(validation_data_dict)
            for res in res_list:
                print res

            current_target_metric = sum(getattr(res, self.target_metric) for res in res_list)
            if (current_target_metric > self.best_target_metric):
                print("highest %s so far. Saving weights." % self.target_metric)
                self.model.save_weights(weights_ofname, overwrite=True)
                self.best_target_metric = current_target_metric
                epochs_no_imporvement = 0
            else:
                epochs_no_imporvement += 1
        # load and return the best model
        print "Loading best model"
        self.model.load_weights(weights_ofname)
        return self
