import sys

import functools
from collections import defaultdict
from random import shuffle, sample

import numpy as np

from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB, 
    encode_peaks_sequence_into_binary_array, 
    PeaksAndLabels,
    build_peaks_label_mat, iter_summit_centered_peaks)
from pyTFbindtools.DB import load_genome_metadata
from pyTFbindtools.cross_validation import ClassificationResult

from fit_selex import SelexDBConn, load_fastq, optional_gzip_open
from pyDNAbinding.binding_model import ConvolutionalDNABindingModel
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ, R, T
from pyDNAbinding.plot import plot_bases, pyplot
from pyDNAbinding.sequence import one_hot_encode_sequence

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda, Layer, Dropout, Permute
from keras import initializations
from keras import backend as K
from keras.models import model_from_yaml

from keras.models import Graph, Sequential
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.layers.core import Dense, Activation, Flatten, Merge, TimeDistributedDense, Reshape

import theano
import theano.tensor as TT

class SeqsTooShort(Exception):
    pass

class NoBackgroundSequences(Exception):
    pass

def expected_F1_loss(y_true, y_pred, beta=0.1):
    min_label = TT.min(y_true)
    max_label = TT.max(y_true)
    y_true = (y_true - min_label)/(max_label - min_label)
    
    expected_true_positives = TT.sum(y_pred*y_true)
    expected_false_positives = TT.sum(y_pred*(1-y_true))
    expected_false_negatives = TT.sum((1-y_pred)*y_true)

    precision = expected_true_positives/(
        expected_true_positives + expected_false_positives + 1.0)
    recall = expected_true_positives/(
        expected_true_positives + expected_false_negatives + 1.0)
    rv = (-1e-6 -(1+beta*beta)*precision*recall)/(beta*beta*precision+recall+2e-6)
    mse = K.mean((1-y_true)*y_pred) - K.mean(y_true*y_pred)
    return rv

def mse_skip_ambig(y_true, y_pred):
    #clipped_pred = K.clip(y_pred, 0.01, 0.99)
    #loss = K.sum(y_true*K.log(clipped_pred))/K.sum(y_true)
    #return loss
    #return K.mean((1-y_true)*y_pred) - K.mean(y_true*y_pred)
    #return K.mean(K.square(y_pred*(y_true > -0.5) - y_true*(y_true > -0.5)), axis=-1)
    return K.mean(K.square(y_pred*(y_true > -0.5) - y_true*(y_true > -0.5)), axis=-1)

def softmax(x, axis):
    scale_factor = x.max(axis=axis, keepdims=True)
    e_x = TT.exp(x - scale_factor)
    weights = e_x/e_x.sum(axis=axis, keepdims=True)
    return (x*weights).sum(axis)

def theano_calc_log_occs(affinities, chem_pot):
    inner = (-chem_pot+affinities)/(R*T)
    lower = TT.switch(inner<-10, TT.exp(inner), 0)
    mid = TT.switch((inner >= -10)&(inner <= 35), 
                    TT.log(1.0 + TT.exp(inner)),
                    0 )
    upper = TT.switch(inner>35, inner, 0)
    return -(lower + mid + upper)

def theano_logistic(affinities, chem_pot):
    return 1/(1+TT.exp(affinities + chem_pot))

def theano_log_sum_log_occs(log_occs):
    # theano.printing.Print('scale_factor')
    scale_factor = (
        TT.max(log_occs, axis=1, keepdims=True))
    centered_log_occs = (log_occs - scale_factor)
    centered_rv = TT.log(TT.sum(TT.exp(centered_log_occs), axis=1))
    return centered_rv + scale_factor.flatten()

import keras.callbacks
class UpdateSampleWeights(keras.callbacks.Callback):
    def __init__(self, sample_weights, *args, **kwargs):
        self._sample_weights = sample_weights
        super(UpdateSampleWeights, self).__init__(*args, **kwargs)
    def on_epoch_end(self, *args, **kwargs):
        print self.my_weights.get_value()
        self.my_weights.set_value(self.my_weights.get_value()/2)

class ConvolutionDNAShapeBinding(Layer):
    def __init__(
            self, nb_motifs, motif_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            initializations.get(init)(x), 
            K.zeros((self.nb_motifs,)) 
        )
        super(ConvolutionDNAShapeBinding, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_motifs': self.nb_motifs,
                  'motif_len': self.motif_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionDNAShapeBinding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self):
        input_dim = self.input_shape[1]
        self.W_shape = (
            self.nb_motifs, 1, self.motif_len, 6)
        self.W, self.b = self.init(self.W_shape)
        self.params = [self.W, self.b]
    
    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                self.nb_motifs,
                # sequence length minus the motif length
                self.input_shape[2], #-self.motif_len+1,
                2)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        fwd_rv = K.conv2d(X[:,:,:,:6], 
                          self.W, 
                          border_mode='valid') \
                 + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X[:,:,:,-6:][:,::-1,::-1], 
                         self.W, 
                         border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        return K.concatenate((fwd_rv, rc_rv), axis=3)
        #return rc_rv


class ConvolutionDNASequenceBinding(Layer):
    def __init__(
            self,
            nb_motifs,
            motif_len, 
            use_three_base_encoding=True,
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        self.use_three_base_encoding = use_three_base_encoding
        self.kwargs = kwargs
        
        self.W = None
        self.b = None
        
        if isinstance(init, ConvolutionalDNABindingModel):
            self.init = lambda x: (
                K.variable(-init.ddg_array[None,None,:,:]), 
                K.variable(np.array([-init.ref_energy,])[:,None]) 
            )
        else:
            self.init = lambda x: (
                initializations.get(init)(x), 
                K.zeros((self.nb_motifs,)) 
            )
        super(ConvolutionDNASequenceBinding, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_motifs': self.nb_motifs,
                  'motif_len': self.motif_len, 
                  'use_three_base_encoding': self.use_three_base_encoding,
                  'init': self.init.__name__}
        base_config = super(ConvolutionDNASequenceBinding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_clone(self):
        """
        
        """
        rv = type(self)(
            self.nb_motifs, 
            self.motif_len, 
            self.use_three_base_encoding, 
            self.init,
            **self.kwargs)
        rv.W = self.W
        rv.b = self.b
        rv.W_shape = self.W_shape
        return rv

    def init_filters(self):
        if self.use_three_base_encoding:
            self.W_shape = (
                self.nb_motifs, 1, self.motif_len, 3)
        else:
            self.W_shape = (
                self.nb_motifs, 1, self.motif_len, 4)
        self.W, self.b = self.init(self.W_shape)
        return

    def build(self):
        #print "Small Domains W SHAPE:", self.W_shape
        assert self.input_shape[3]==4, "Expecting one-hot-encoded DNA sequence."
        if self.W is None:
            assert self.b is None
            self.init_filters()
        self.params = [self.W[0], self.b[0]]
    
    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0],
                2,
                # sequence length minus the motif length
                self.input_shape[2]-self.motif_len+1,
                self.nb_motifs)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        if self.use_three_base_encoding:
            X_fwd = X[:,:,:,1:]
            X_rc = X[:,:,:,:3]
        else:
            X_fwd = X
            X_rc = X

        if self.W[1] is not None:
            W = self.W[0][self.W[1],:,:,:]
        else:
            W = self.W[0]
        if self.b[1] is not None:
            b = self.b[0][self.b[1]]
        else:
            b = self.b[0]
        
        fwd_rv = K.conv2d(X_fwd, W, border_mode='valid') \
                 + K.reshape(b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X_rc, W[:,:,::-1,::-1], border_mode='valid') \
                + K.reshape(b, (1, self.nb_motifs, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=3)            
        #return rv.dimshuffle((0,3,2,1))
        return K.permute_dimensions(rv, (0,3,2,1))

class ConvolutionBindingSubDomains(Layer):
    def __init__(
            self,
            nb_domains, 
            domain_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_domains = nb_domains
        self.domain_len = domain_len

        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            (initializations.get(init)(x), None),
            (K.zeros((self.nb_domains,)), None) 
        )
        self.kwargs = kwargs
        
        self.W_shape = None
        self.W = None
        self.b = None
        
        super(ConvolutionBindingSubDomains, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_domains': self.nb_domains,
                  'domain_len': self.domain_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionBindingSubDomains, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def create_clone(self):
        """
        
        """
        rv = type(self)(
            self.nb_domains, 
            self.domain_len, 
            self.init,
            **self.kwargs)
        rv.W = self.W
        rv.b = self.b
        rv.W_shape = self.W_shape
        return rv

    def init_filters(self, num_input_filters):
        self.W_shape = (
            self.nb_domains, 1, self.domain_len, num_input_filters)
        self.W, self.b = self.init(self.W_shape)
        return

    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        assert self.input_shape[1] == 2
        if self.W is None:
            assert self.b is None
            self.init_filters(self.input_shape[1])
        #print "Subdomains W SHAPE:", self.W_shape
        #assert self.input_shape[3] == self.W_shape[3]
        self.params = [self.W[0], self.b[0]]
    
    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                2,
                # sequence length minus the motif length
                self.input_shape[2]-self.domain_len+1,
                self.nb_domains)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        if self.W[1] is not None:
            W = self.W[0][self.W[1],:,:,:]
        else:
            W = self.W[0]
        if self.b[1] is not None:
            b = self.b[0][self.b[1]]
        else:
            b = self.b[0]
        fwd_rv = K.conv2d(X[:,0:1,:,:], W, border_mode='valid')  \
                 + K.reshape(b, (1, self.nb_domains, 1, 1))
        # # [:,:,::-1,::-1]
        rc_rv = K.conv2d(X[:,1:2,:,:], W[:,:,::-1,:], border_mode='valid') \
                + K.reshape(b, (1, self.nb_domains, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=3)
        #return rv.dimshuffle((0,3,2,1))
        return K.permute_dimensions(rv, (0,3,2,1))

class NormalizedOccupancy(Layer):
    def __init__(self, **kwargs):
        self.input = K.placeholder(ndim=4)        
        super(NormalizedOccupancy, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(NormalizedOccupancy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        #assert self.input_shape[3] == 2
        self.chem_affinity = K.variable(0.0)
        self.params = [self.chem_affinity]
    
    @property
    def output_shape(self):
        #return (self.input_shape[0],) # 1, 1, 1)
        return (# number of obseravations
                self.input_shape[0], 
                1,
                # sequence length minus the motif length
                self.input_shape[2], #-self.domain_len+1,
                2*self.input_shape[3])
    
    def get_output(self, train=False):
        X = self.get_input(train)
        rv = theano_logistic(-X, self.chem_affinity)
        #rv = TT.exp(theano_calc_log_occs(-X, self.chem_affinity))
        return K.concatenate([rv[:,0:1,:,:], rv[:,1:2,:,:]], axis=3)
        #p_op = theano.printing.Print("="*40)
        log_occs = TT.flatten(
            theano_calc_log_occs(X, self.chem_affinity), outdim=2)
        return K.exp(theano_calc_log_occs(X, self.chem_affinity))
        scale_factor = K.max(log_occs, axis=1)
        centered_log_occs = log_occs - K.reshape(scale_factor, (X.shape[0],1))
        norm_occs = K.sum(K.exp(centered_log_occs), axis=1)
        log_sum_log_occs = TT.reshape(K.log(norm_occs) + scale_factor, (X.shape[0],1))
        return K.flatten(K.exp(log_sum_log_occs))
        norm_log_occs = log_occs-log_sum_log_occs
        norm_occs = K.exp(norm_log_occs)
        return K.reshape(norm_occs, X.shape)
        return K.max(norm_occs, axis=1) #K.reshape(norm_occs, X.shape)


class MonitorAccuracy(keras.callbacks.Callback):    
    def on_epoch_end(self, *args, **kwargs):
        val_results = {}
        data = self.model.named_validation_inputs
        proba = self.model.predict(data)
        for key, prbs in proba.iteritems():
            print prbs.shape
            if prbs.shape[1] == 1:
                prbs_subset = prbs.ravel()
                classes = np.array(prbs_subset > 0.5, dtype=int)
                truth = data[key].ravel()
                res = ClassificationResult(truth, classes, prbs_subset)
                val_results[key] = res
            else:
                for i in xrange(prbs.shape[1]):
                    prbs_subset = prbs[:,i].ravel()
                    classes = np.array(prbs_subset > 0.5, dtype=int)
                    truth = data[key][:,i].ravel()
                    res = ClassificationResult(truth, classes, prbs_subset)
                    val_results[key + "_%s" % self.model.tf_names[i]] = res

        for key, res in sorted(val_results.iteritems()):
            print key.ljust(40), res

def calc_accuracy(pred_probs, labels):
    return float((pred_probs.round() == labels[:,None]).sum())/labels.shape[0]

def upsample(seqs, num_seqs):
    new_seqs = []
    new_seqs.extend(seqs)
    while len(new_seqs) < num_seqs:
        new_seqs.extend(seqs[:num_seqs-len(new_seqs)])
    return new_seqs

class JointBindingModel(Graph):
    def calc_accuracy(self, val=True):
        if val == True:
            inputs = self.named_validation_inputs
        else:
            inputs = self.named_inputs

        pred_prbs = self.predict(inputs)
        acc_res = {}
        for factor_key in pred_prbs.iterkeys():
            acc_res[factor_key] = calc_accuracy(
                pred_prbs[factor_key], inputs[factor_key])
        return acc_res

    def __init__(self, num_samples, tf_names, *args, **kwargs):
        # store a dictionary of round indexed selex sequences, 
        # indexed by experiment ID
        self.num_samples = num_samples
        assert self.num_samples%2 == 0
        self.tf_names = tf_names
        
        self.selex_seqs = defaultdict(dict)
        self.named_validation_inputs = {}
        self.named_inputs = {}
        self.named_losses = {}
        self.sample_weights = {}
        
        # initialize the base convolutional filter
        self.num_small_convs = 64
        self.small_conv_size = 4
        self.short_conv_layer = ConvolutionDNASequenceBinding(
            self.num_small_convs, # numConv
            self.small_conv_size, # convWidth, 
            #init=initial_model
        )
        self.short_conv_layer.init_filters()

        # initialize the full subdomain convolutional filter
        self.num_binding_subdomain_convs = 8
        self.num_affinity_outputs = self.num_binding_subdomain_convs*len(tf_names)
        self.binding_subdomain_conv_size = 9
        self.stacked_subdomains_layer_shape = (
            (0+self.num_binding_subdomain_convs)*len(tf_names), 
            1, 
            self.binding_subdomain_conv_size, 
            3) # self.num_small_convs)
        self.stacked_subdomains_layer_filters = (
            self.stacked_subdomains_layer_shape,
            K.zeros((self.stacked_subdomains_layer_shape[0],)),
            initializations.get('glorot_uniform')(
                self.stacked_subdomains_layer_shape)
        )

        # extract the tf specific subtensors
        self.binding_subdomains_layer = ConvolutionDNASequenceBinding( #
                (0+self.num_binding_subdomain_convs)*len(tf_names), # numConv
                self.binding_subdomain_conv_size, # convWidth,
        )
        self.binding_subdomains_layer.W_shape = (
            self.stacked_subdomains_layer_filters[0] )
        self.binding_subdomains_layer.b = (
            self.stacked_subdomains_layer_filters[1], None)
        self.binding_subdomains_layer.W = (
            self.stacked_subdomains_layer_filters[2], None)

        self.binding_subdomain_layers = {}
        for i, tf_name in enumerate(tf_names):
            subtensor_slice = slice(
                i*self.num_binding_subdomain_convs,
                (i+1)*self.num_binding_subdomain_convs)
            W_shape = list(self.stacked_subdomains_layer_shape)
            W_shape[0] = self.num_binding_subdomain_convs
            W_shape = tuple(W_shape)
            b = self.stacked_subdomains_layer_filters[1] #subtensor_slice]
            W = self.stacked_subdomains_layer_filters[2] #[subtensor_slice]
            binding_sub_domains_layer = ConvolutionDNASequenceBinding(
                self.num_binding_subdomain_convs, # numConv
                self.binding_subdomain_conv_size, # convWidth,
            )
            binding_sub_domains_layer.W_shape = W_shape
            binding_sub_domains_layer.W = (W, subtensor_slice)
            binding_sub_domains_layer.b = (b, subtensor_slice)
            self.binding_subdomain_layers[tf_name] = binding_sub_domains_layer

        super(Graph, self).__init__(*args, **kwargs)

    def add_affinity_layers(self, input_name, output_name, tf_name=None):
        """
        if True:
            short_conv_layer = ConvolutionDNASequenceBinding(
                self.num_small_convs, # numConv
                self.small_conv_size, # convWidth, 
                #init=initial_model
            )
        else:
            short_conv_layer = self.short_conv_layer.create_clone()
        
        self.add_node(
            short_conv_layer,
            input=input_name,
            name=input_name + ".1"
        )
        """

        subdomain_input_name = input_name # + ".1"
        if tf_name is None:
            self.add_node(
                self.binding_subdomains_layer.create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        else:
            assert tf_name in self.binding_subdomain_layers
            self.add_node(
                self.binding_subdomain_layers[tf_name].create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        return
    
    def add_invitro_layer(self, tf_name, factor_id, selex_exp_id, 
                          fwd_seqs, shape_seqs, labels):
        name_prefix = 'invitro_%s_%s_%s_' % (
            tf_name, factor_id, selex_exp_id)

        self.add_input(
            name=name_prefix+'one_hot_sequence', 
            input_shape=(None, fwd_seqs.shape[2], 4))
        assert name_prefix+'one_hot_sequence' not in self.named_inputs
        self.named_inputs[name_prefix+'one_hot_sequence'] = fwd_seqs
        self.add_affinity_layers(
            input_name=name_prefix+'one_hot_sequence', 
            output_name=name_prefix + 'binding_affinities',
            tf_name=tf_name)
        
        self.add_node(
            NormalizedOccupancy(),
            name=name_prefix + 'occupancies',
            input=name_prefix + 'binding_affinities')

        #self.add_node(
        #    Activation('relu'),
        #    name=name_prefix + 'occupancies',
        #    input=name_prefix + 'binding_affinities')

        self.add_node(
            Lambda(lambda x: K.max(K.batch_flatten(x), axis=1, keepdims=True)),
            name=name_prefix+'max', 
            input=name_prefix+'occupancies')

        self.add_output(name=name_prefix+'output', input=name_prefix+'max')
        
        assert name_prefix + 'output' not in self.named_inputs
        self.named_inputs[name_prefix + 'output'] = labels

        self.named_losses[name_prefix + 'output'] = 'mse' #mse_skip_ambig #expected_F1_loss
        #expected_F1_loss #'mse' #my_mean_squared_error

        return

    def add_invivo_layer(self, name_prefix, seq_length, output_dim):
        self.add_input(
            name=name_prefix+'one_hot_sequence', 
            input_shape=(1, seq_length, 4))

        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly

        self.add_affinity_layers(
            input_name=name_prefix+'one_hot_sequence', 
            output_name=name_prefix+'binding_affinities.1')

        self.add_node(
            ConvolutionBindingSubDomains(
                nb_domains=8, 
                domain_len=32,
            ),
            name=name_prefix + 'binding_affinities.2',
            input=name_prefix + 'binding_affinities.1'
        )

        self.add_node(
            NormalizedOccupancy(),
            name=name_prefix + 'occupancies.1',
            input=name_prefix + 'binding_affinities.2')

        
        #self.add_node(
        #    Lambda(lambda x: K.max(K.batch_flatten(x), axis=1, keepdims=True)),
        #    name=name_prefix+'max', 
        #    input=name_prefix+'occupancies.1')

        self.add_node(
            Flatten(),
            input=name_prefix+'occupancies.1',
            name=name_prefix+'occupancies.2'
        )
        
        self.add_node(
            Dense(64, activation='sigmoid'), # numOutputNodes
            input=name_prefix+'occupancies.2',
            name=name_prefix+'dense1'            
        ) 
        self.add_node(
            Dropout(0.5),
            name=name_prefix + 'dense2',
            input=name_prefix + 'dense1')

        self.add_node(
            Dense(output_dim, activation='sigmoid'), # numOutputNodes
            input=name_prefix+'dense2',
            name=name_prefix+'dense'            
        )
        
        self.add_output(
            name=name_prefix+'output', 
            input=name_prefix+'dense')
    
    def add_chipseq_sample(self, 
                           peaks_and_labels, 
                           genome_fasta, 
                           tf_name, 
                           sample_id):
        train_start_index = int(self.num_samples*0.1)
        print "Adding ChIP-seq data for %s (sample ID %s)" % (
                tf_name, sample_id)
        name_prefix = 'invivo_%s_%s_' % (tf_name, sample_id) 
        fwd_seqs = encode_peaks_sequence_into_binary_array(
            [pk.peak for pk in peaks_and_labels], genome_fasta)[:,None,:,:]
        labels = np.array([pk.label for pk in peaks_and_labels])
        print round(float(labels.sum())/labels.shape[0], 3)
        self.add_invivo_layer(
            name_prefix,
            fwd_seqs.shape[2],
            1)

        assert name_prefix+'one_hot_sequence' not in self.named_inputs
        self.named_inputs[name_prefix+'one_hot_sequence'] = fwd_seqs[train_start_index:]
        assert name_prefix + 'output' not in self.named_inputs
        self.named_inputs[name_prefix + 'output'] = labels[train_start_index:]
        weights = np.ones_like(labels[train_start_index:])
        self.sample_weights[name_prefix + 'output'] = weights.ravel()
        self.named_losses[name_prefix + 'output'] = 'mse' # mse_skip_ambig #expected_F1_loss

        self.named_validation_inputs[
            name_prefix+'one_hot_sequence'] = fwd_seqs[:train_start_index,:,:,:]
        self.named_validation_inputs[name_prefix+'output'] = labels[
            :train_start_index]
        return

    def add_chipseq_layer(self, tf_id, tf_name):
        genome_fasta = FastaFile(
            load_genome_metadata(1).filename) # annotation_id 
        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            tf_id, # args.tf_id,
            1, #args.annotation_id,
            500, # args.half_peak_width, 
            self.num_samples, #args.max_num_peaks_per_sample, 
            include_ambiguous_peaks=True,
            order_by_accessibility=True).remove_ambiguous_labeled_entries()
        peaks = PeaksAndLabels(sample(peaks_and_labels, self.num_samples))
        self.add_chipseq_sample(peaks, genome_fasta, tf_name, 'top')
        return
       
        # group peaks by their sample
        sample_grpd_peaks = defaultdict(list)
        assert False
        sample_grpd_peaks[pk.sample].append(pk)
        peaks_seqs, peaks_labels = {}, {}
        for sample_peaks in sample_grpd_peaks.itervalues():
            shuffle(sample_peaks)
        for sample_id, sample_pks in sample_grpd_peaks.iteritems():
            self.add_chipseq_sample(sample_pks, genome_fasta, tf_name, sample_id)
        return

    def add_chipseq_samples(self, pks_and_labels): 
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_id
        name_prefix = 'invivo_%s_' % pks_and_labels.sample_id 

        #print (pks_and_labels.labels.sum(axis=1)
        #         )/pks_and_labels.labels.shape[0]).round(3)
        self.add_invivo_layer(
            name_prefix, 
            pks_and_labels.fwd_seqs.shape[2], 
            pks_and_labels.labels.shape[1])

        assert name_prefix+'one_hot_sequence' not in self.named_inputs
        self.named_inputs[name_prefix+'one_hot_sequence'] = pks_and_labels.train_fwd_seqs
        assert name_prefix + 'output' not in self.named_inputs
        self.named_inputs[name_prefix + 'output'] = pks_and_labels.train_labels
        num_zeros = (pks_and_labels.train_labels == 0.0).sum()
        num_ones = (pks_and_labels.train_labels == 1.0).sum()
        weights = np.zeros_like(pks_and_labels.train_labels)
        weights[pks_and_labels.train_labels == 0.0] = (1 - num_zeros/float(num_ones + num_zeros))
        weights[pks_and_labels.train_labels == 1.0] = (1 - num_ones/float(num_ones + num_zeros))
        self.sample_weights[name_prefix + 'output'] = weights.ravel()
        print weights
        self.named_losses[name_prefix + 'output'] = 'mse' # mse_skip_ambig #expected_F1_loss

        self.named_validation_inputs[
            name_prefix+'one_hot_sequence'] = pks_and_labels.validation_fwd_seqs
        self.named_validation_inputs[name_prefix+'output'] = pks_and_labels.validation_labels

        return

    def add_selex_layer(self, selex_exp_id):
        # load connect to the DB, and find the factor name
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', selex_exp_id)
        factor_name = selex_db_conn.get_factor_name()
        factor_id = selex_db_conn.get_factor_id()

        # load the sequencess
        print "Loading sequences for %s-%s (exp ID %i)" % (
                factor_name, factor_id, selex_exp_id)
        fnames, bg_fname = selex_db_conn.get_fnames()
        if bg_fname is None:
            raise NoBackgroundSequences("No background sequences for %s (%i)." % (factor_name, selex_exp_id))

        with optional_gzip_open(fnames[max(fnames.keys())]) as fp:
            bnd_seqs = load_fastq(fp, self.num_samples/2)
        with optional_gzip_open(bg_fname) as fp:
            unbnd_seqs = load_fastq(fp, self.num_samples/2)
        if len(bnd_seqs[0]) < 20:
            raise SeqsTooShort("Seqs too short for %s (exp %i)." % (factor_name, selex_exp_id))
        
        if len(unbnd_seqs) < self.num_samples/2:
            unbnd_seqs = upsample(unbnd_seqs, self.num_samples/2)
        if len(bnd_seqs) < self.num_samples/2:
            bnd_seqs = upsample(bnd_seqs, self.num_samples/2)

        train_start_index = int(self.num_samples*0.1)
        fwd_seqs, shape_seqs, labels = get_coded_seqs_and_labels(
            unbnd_seqs, bnd_seqs)
        
        self.add_invitro_layer(
            factor_name,
            factor_id,
            selex_exp_id,
            fwd_seqs[train_start_index:,:,:,:], 
            None, 
            labels[train_start_index:])

        name_prefix = 'invitro_%s_%s_%s_' % (
            factor_name, factor_id, selex_exp_id)
        self.named_validation_inputs[
            name_prefix+'one_hot_sequence'] = fwd_seqs[:train_start_index,:,:,:]
        if shape_seqs is not None:
            self.named_validation_inputs[
                name_prefix+'shape'] = shape_seqs[:train_start_index,:,:,:]
        self.named_validation_inputs[name_prefix+'output'] = labels[
            :train_start_index]
    
    def compile(self, *args, **kwargs):
        compile = functools.partial(
            super(JointBindingModel, self).compile,
            loss=self.named_losses,
            optimizer=Adam(),
            sample_weight_modes=dict((key, None) for key in self.sample_weights)
        )
        return compile(*args, **kwargs)

    def fit(self, 
            data=None, 
            validation_split=0.1, 
            batch_size=100, 
            nb_epoch=100,
            *args, **kwargs):
        if data is None: data = self.named_inputs
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, verbose=1, mode='auto')
        monitor_accuracy_cb = MonitorAccuracy()

        #print data.keys()
        #pass

        for key, val in data.iteritems():
            print key, val.shape
        #assert False

        fit = functools.partial(
            super(JointBindingModel, self).fit,
            data=data,
            validation_split=validation_split,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[monitor_accuracy_cb, early_stop],
            shuffle=True,
            sample_weight=self.sample_weights
        )
        return fit(*args, **kwargs)

def get_coded_seqs_and_labels(unbnd_seqs, bnd_seqs):
    num_seqs = min(len(unbnd_seqs), len(bnd_seqs))
    unbnd_seqs = FixedLengthDNASequences(unbnd_seqs[:num_seqs], include_shape=False)
    bnd_seqs = FixedLengthDNASequences(bnd_seqs[:num_seqs], include_shape=False)
    
    permutation = np.random.permutation(2*num_seqs)
    fwd_one_hot_seqs = np.vstack(
        (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs)
    )[permutation,None,:,:]

    if unbnd_seqs.fwd_shape_features is None:
        shape_seqs = None
    else:
        fwd_shape_seqs = np.vstack(
            (unbnd_seqs.fwd_shape_features, bnd_seqs.fwd_shape_features)
        )[permutation,:,:]
        rc_shape_seqs = np.vstack(
            (unbnd_seqs.rc_shape_features, bnd_seqs.rc_shape_features)
        )[permutation,:,:]
        shape_seqs = np.concatenate((fwd_shape_seqs, rc_shape_seqs), axis=2)[:,None,:,:]

    labels = np.hstack(
        (np.zeros(len(unbnd_seqs), dtype='float32'), 
         np.ones(len(bnd_seqs), dtype='float32'))
    )[permutation]
    return fwd_one_hot_seqs, shape_seqs, labels # rc_one_hot_seqs, 

import pandas as pd

class SamplePeaksAndLabels():
    def one_hot_code_peaks_sequence(self):
        pk_width = self.pks[0][2] - self.pks[0][1]
        rv = 0.25 * np.ones((len(self.pks), pk_width, 4), dtype='float32')
        for i, data in enumerate(self.pks):
            assert pk_width == data[2] - data[1]
            seq = self.genome_fasta.fetch(str(data[0]), data[1], data[2])
            if len(seq) != pk_width: continue
            coded_seq = one_hot_encode_sequence(seq)
            rv[i,:,:] = coded_seq
        return rv[:,None,:,:]

    def _init_pks_and_label_mats(self):
        from pyTFbindtools.DB import load_tf_names
        # load the matrix. This is probably cached on disk, but may need to 
        # be recreated
        (self.pks, self.tf_ids, (self.idr_optimal_labels, self.relaxed_labels)
            ) = build_peaks_label_mat(
                self.annotation_id, self.sample_id, self.half_peak_width)
        self.pk_record_type = type(self.pks[0])
        pk_types = ('S64', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S')
        self.pks = np.array(self.pks, dtype=zip(self.pks[0]._fields, pk_types))
        # sort the peaks by accessibility
        index = np.lexsort((self.pks['start'], -self.pks['signalValue']))
        self.pks = self.pks[index]
        self.idr_optimal_labels = self.idr_optimal_labels[index,:]
        self.relaxed_labels = self.relaxed_labels[index,:]

        # keep the top n_samples peaks and max_num_tfs tfs
        print "Loading tf names"
        all_tf_names = load_tf_names(self.tf_ids)
        # make sure all of the tfnames actually exist
        assert ( self.desired_tf_names is None 
                 or all(tf_name in all_tf_names 
                        for tf_name in self.desired_tf_names) 
        )
        # filter the tf set 
        filtered_tfs = sorted([
            (tf_name, tf_id, i) for i, (tf_name, tf_id) 
            in enumerate(zip(all_tf_names, self.tf_ids)) 
            if tf_name in self.desired_tf_names
        ])
        self.tf_names, self.tf_ids, tf_indices = zip(*filtered_tfs)
        self.pks = self.pks[:self.n_samples]
        self.idr_optimal_labels = self.idr_optimal_labels[
            :self.n_samples, np.array(tf_indices)]
        self.relaxed_labels = self.relaxed_labels[
            :self.n_samples, np.array(tf_indices)]
        
        # set the ambiguous labels to -1
        self.ambiguous_pk_indices = (
            self.idr_optimal_labels != self.relaxed_labels)
        self.clean_labels = self.idr_optimal_labels.copy()
        self.clean_labels[self.ambiguous_pk_indices] = -1

        # print out balance statistics
        #print (self.idr_optimal_labels.sum(axis=0)/self.idr_optimal_labels.shape[0])
        #print (self.relaxed_labels.sum(axis=0)/self.relaxed_labels.shape[0])
        #print (self.ambiguous_pk_indices.sum(axis=0)/float(self.ambiguous_pk_indices.shape[0]))

        return
    
    def __init__(
            self, sample_id, n_samples, annotation_id=1, 
            half_peak_width=500, tf_names=None):
        self.sample_id = sample_id
        self.annotation_id = annotation_id
        self.half_peak_width = half_peak_width
        self.n_samples = n_samples
        self.desired_tf_names = tf_names
        self._init_pks_and_label_mats()

        # initialize the training and validation data
        index = np.argsort(np.random.random(len(self.pks)))
        self.pks = self.pks[index]
        self.idr_optimal_labels = self.idr_optimal_labels[index,:]
        self.relaxed_labels = self.relaxed_labels[index,:]
        self.ambiguous_pk_indices = self.ambiguous_pk_indices[index,:]
        self.ambiguous_labels = self.clean_labels[index,:]
        from scipy.stats import itemfreq
        print "IDR Optimal"
        print itemfreq(self.idr_optimal_labels).T
        print "Relaxed"
        print itemfreq(self.relaxed_labels).T
        print "Ambiguous"
        print itemfreq(self.ambiguous_labels).T
        print 
        training_index = int(self.n_samples*0.1)        
        self.labels = self.idr_optimal_labels # relaxed_labels
        self.train_labels = self.labels[training_index:]
        self.validation_labels = self.labels[:training_index]
        
        # code the peaks' sequence
        print "Coding peaks"
        self.genome_fasta = FastaFile(
            load_genome_metadata(self.annotation_id).filename)
        self.fwd_seqs = self.one_hot_code_peaks_sequence()
        self.train_fwd_seqs = self.fwd_seqs[training_index:]
        self.validation_fwd_seqs = self.fwd_seqs[:training_index]
        
        return
    

from pyTFbindtools.peaks import build_peaks_label_mat
def main():
    """
    peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        'T044268_1.02', # args.tf_id,
        1, #args.annotation_id,
        500, # args.half_peak_width, 
        10000, #args.max_num_peaks_per_sample, 
        include_ambiguous_peaks=True,
        order_by_accessibility=True)#.remove_ambiguous_labeled_entries()
    counts = defaultdict(int)
    counts = defaultdict(int)
    for pk in peaks_and_labels:
        counts[(pk.sample, pk.label)] += 1
    for x in sorted(counts.items()):
        print x
    assert False
    """
    n_samples = 25000
    sample_id = 'E123'
    sample_peaks_and_labels = SamplePeaksAndLabels(
        sample_id, n_samples, half_peak_width=500, tf_names=['MAX',])
    model = JointBindingModel(n_samples, sample_peaks_and_labels.tf_names)
    print sample_peaks_and_labels.tf_names
    model.add_selex_layer(441) # MAX
    model.add_chipseq_layer('T011266_1.02', 'MAX') 
    #model.add_chipseq_samples(sample_peaks_and_labels)
    print "Compiling Model"
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=100)
    return

    """
    model.add_selex_layer(441) # MAX
    model.add_selex_layer(202) # YY1
    model.add_chipseq_layer('T011266_1.02', 'MAX') 
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=5)
    return
    """

    import psycopg2
    conn = psycopg2.connect(
        "host=%s dbname=%s user=%s" % ('mitra', 'cisbp', 'nboley'))
    cur = conn.cursor()
    query = """
      SELECT tfs.family_id, family_name, array_agg(tfs.tf_name) 
        FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN tf_families 
    GROUP BY family_id, family_name;
    """ # group tfs by their families
    
    query = """
      SELECT selex_experiments.tf_id, tfs.tf_name, array_agg(distinct selex_exp_id)
        FROM roadmap_matched_chipseq_peaks NATURAL JOIN selex_experiments NATURAL JOIN tfs 
    GROUP BY selex_experiments.tf_id, tfs.tf_name
    ORDER BY tf_name;
    """
    cur.execute(query)
    # filter out the TFs that dont have background sequence or arent long enough
    res = [ x for x in cur.fetchall() 
            if x[1] not in ('E2F1', 'E2F4', 'POU2F2', 'PRDM1', 'RXRA')
            #and x[1] in ('MAX',) # 'YY1', 'RFX5', 'USF', 'PU1', 'CTCF')
    ][:25]
    
    tf_names = [x[1] for x in res]
    print tf_names
    model = JointBindingModel(n_samples, tf_names)
    
    for tf_id, tf_name, selex_exp_ids in res:
        for selex_exp_id in selex_exp_ids:
            try: 
                model.add_selex_layer(selex_exp_id)
            except SeqsTooShort, inst:
                print inst
                continue
            except NoBackgroundSequences, inst:
                print inst
                continue
            except Exception, inst:
                raise

    model.add_chipseq_layer('T011266_1.02', 'MAX') 
    #model.add_chipseq_layer('T044261_1.02', 'YY1')

    #for tf_id, tf_name, selex_exp_ids in res:
    #    try:
    #        model.add_chipseq_layer(tf_id, tf_name)
    #    except Exception, inst:
    #        print "Couldnt load ChIP-seq data for %s" % tf_name
    #        continue
    #    else:
    #        break
    
    #model.compile()
    #model.fit(validation_split=0.1, batch_size=100, nb_epoch=5)
    #model.add_chipseq_layer('T044261_1.02', 'YY1')
    #model.add_chipseq_layer('T011266_1.02', 'MAX') 
    #model.add_chipseq_layer('T025286_1.02', 'JUND')
    #model.add_chipseq_layer('T044306_1.02', 'EGR1')
    #model.add_chipseq_layer('T014210_1.02', 'MYC')
    #model.add_chipseq_layer('T044268_1.02', 'CTCF')
    
    print "Compiling Model"
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=100)

    pass
    

main()
