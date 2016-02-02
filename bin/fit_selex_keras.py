import functools
from collections import defaultdict
from random import shuffle

import numpy as np

from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB, 
    encode_peaks_sequence_into_binary_array )
from pyTFbindtools.DB import load_genome_metadata
from pyTFbindtools.cross_validation import ClassificationResult

from fit_selex import SelexDBConn, load_fastq, optional_gzip_open
from pyDNAbinding.binding_model import ConvolutionalDNABindingModel
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ, R, T
from pyDNAbinding.plot import plot_bases, pyplot

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda, Layer, Dropout
from keras import initializations
from keras import backend as K
from keras.models import model_from_yaml

from keras.models import Graph
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.layers.core import Dense, Activation, Flatten, Merge, TimeDistributedDense, Reshape

import theano
import theano.tensor as TT

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

    return (-1e-6 -(1+beta*beta)*precision*recall)/(beta*beta*precision+recall+2e-6)

def my_mean_squared_error(y_true, y_pred):
    #clipped_pred = K.clip(y_pred, 0.01, 0.99)
    #loss = K.sum(y_true*K.log(clipped_pred))/K.sum(y_true)
    #return loss
    return K.mean((1-y_true)*y_pred) - K.mean(y_true*y_pred)
    #return K.mean(K.square(y_pred - y_true), axis=-1)

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
                self.input_shape[2]-self.motif_len+1,
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
        assert self.input_shape[3]==4, "Expecting one-hot-encoded DNA sequence."
        if self.W is None:
            assert self.b is None
            self.init_filters()
        self.params = [self.W, self.b]
    
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
        
        fwd_rv = K.conv2d(X_fwd, self.W, border_mode='valid') \
                 + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X_rc, self.W[:,:,::-1,::-1], border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=3)            
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
            initializations.get(init)(x), 
            K.zeros((self.nb_domains,)) 
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
        print "W SHAPE:", self.W_shape
        assert self.input_shape[3] == self.W_shape[3]
        self.params = [self.W, self.b]
    
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
        fwd_rv = K.conv2d(X[:,0:1,:,:], self.W, border_mode='valid')  \
                 + K.reshape(self.b, (1, self.nb_domains, 1, 1))
        # # [:,:,::-1,::-1]
        rc_rv = K.conv2d(X[:,1:2,:,:], self.W[:,:,::-1,::-1], border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_domains, 1, 1))
        rv = K.concatenate((fwd_rv, rc_rv), axis=3)
        return K.permute_dimensions(rv, (0,3,2,1))

class NormalizedOccupancy(Layer):
    def __init__(
            self, domain_len, chem_affinity=0.0, **kwargs):
        self.domain_len = domain_len
        self.init_chem_affinity = chem_affinity
        self.input = K.placeholder(ndim=4)        
        super(NormalizedOccupancy, self).__init__(**kwargs)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'domain_len': self.domain_len,
                  'init_chem_affinity': self.chem_affinity} 
        base_config = super(NormalizedOccupancy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        #assert self.input_shape[3] == 2
        self.chem_affinity = K.variable(self.init_chem_affinity)
        self.params = [self.chem_affinity]
    
    @property
    def output_shape(self):
        #return (self.input_shape[0],) # 1, 1, 1)
        return (# number of obseravations
                self.input_shape[0], 
                2,
                # sequence length minus the motif length
                self.input_shape[2]-self.domain_len+1,
                self.input_shape[3])
    
    def get_output(self, train=False):
        X = self.get_input(train)
        return TT.exp(theano_calc_log_occs(-X, self.chem_affinity))
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
            prbs = prbs.ravel()
            classes = np.array(prbs > 0.5, dtype=int)
            truth = data[key]
            res = ClassificationResult(truth, classes, prbs)
            val_results[key] = res

        for key, res in sorted(val_results.iteritems()):
            print key.ljust(50), res

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

    def __init__(self, num_samples, *args, **kwargs):
        # store a dictionary of round indexed selex sequences, 
        # indexed by experiment ID
        self.num_samples = num_samples
        assert self.num_samples%2 == 0
        self.selex_seqs = defaultdict(dict)
        self.named_validation_inputs = {}
        self.named_inputs = {}
        self.named_losses = {}
        
        self.num_binding_subdomain_convs = 10
        self.binding_subdomain_layers = {}

        self.num_small_convs = 100
        self.small_conv_size = 7
        self.short_conv_layer = ConvolutionDNASequenceBinding(
            self.num_small_convs, # numConv
            self.small_conv_size, # convWidth, 
            #init=initial_model
        )
        self.short_conv_layer.init_filters()

        self.stacked_subdomains_layer = None

        super(Graph, self).__init__(*args, **kwargs)

    def add_affinity_layers(
            self, 
            tf_name, 
            input_name, output_name, 
            use_all_binding_subdomains=False):
        self.add_node(
            self.short_conv_layer.create_clone(),
            input=input_name,
            name=input_name + ".1"
        )

        if tf_name not in self.binding_subdomain_layers:
            print "Initializing new binding sub-domains layer"
            binding_sub_domains_layer = ConvolutionBindingSubDomains(
                self.num_binding_subdomain_convs, # numConv
                14, # convWidth,
            )
            binding_sub_domains_layer.init_filters(self.num_small_convs)
            self.binding_subdomain_layers[tf_name] = binding_sub_domains_layer

        if use_all_binding_subdomains:
            self.add_node(
                self.build_stacked_subdomains_layer().create_clone(),
                input=input_name + ".1",
                name=output_name
            )
        else:
            self.add_node(
                self.binding_subdomain_layers[tf_name].create_clone(),
                input=input_name + ".1",
                name=output_name
            )
        return

    def build_stacked_subdomains_layer(self):
        if self.stacked_subdomains_layer is not None:
            return self.stacked_subdomains_layer
        tf_names = sorted(self.binding_subdomain_layers.keys())
        Ws = [self.binding_subdomain_layers[tf_name].W for tf_name in tf_names]
        bs = [self.binding_subdomain_layers[tf_name].b for tf_name in tf_names]
        print "w_shapes", [
            self.binding_subdomain_layers[tf_name].W_shape
            for tf_name in tf_names
        ]
        nb_domains = sum(self.binding_subdomain_layers[tf_name].W_shape[1]
                         for tf_name in tf_names)
        domain_len = next(self.binding_subdomain_layers.itervalues()).W_shape[2]
        assert all(domain.W_shape[2] == domain_len 
                   for domain in self.binding_subdomain_layers.itervalues())
        W_shape = (None, nb_domains, domain_len, 2)
        self.binding_subdomain_convs = [
            W_shape, K.concatenate(bs, axis=0), K.concatenate(Ws, axis=1)]
        self.stacked_subdomains_layer = ConvolutionBindingSubDomains(
            nb_domains=nb_domains, domain_len=domain_len)
        self.stacked_subdomains_layer.W_shape = self.binding_subdomain_convs[0]
        self.stacked_subdomains_layer.b = self.binding_subdomain_convs[1]
        self.stacked_subdomains_layer.W = self.binding_subdomain_convs[2]
        return self.stacked_subdomains_layer

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
            tf_name, 
            input_name=name_prefix+'one_hot_sequence', 
            output_name=name_prefix + 'binding_affinities')
        
        self.add_node(
            NormalizedOccupancy(1),
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

        self.named_losses[name_prefix + 'output'] = 'mse'
        #expected_F1_loss #'mse' #my_mean_squared_error

        return

    def add_invivo_layer(self, tf_name, sample_id, fwd_seqs, labels):
        name_prefix = 'invivo_%s_%s_' % (tf_name, sample_id) 

        self.add_input(
            name=name_prefix+'one_hot_sequence', 
            input_shape=(1, fwd_seqs.shape[2], 4))
        assert name_prefix+'one_hot_sequence' not in self.named_inputs
        self.named_inputs[name_prefix+'one_hot_sequence'] = fwd_seqs

        seq_len = fwd_seqs.shape[2]
        numConv = 30
        convStack = 1
        convWidth = 4
        convHeight = 45
        maxPoolSize = 20
        maxPoolStride = 20
        numConvOutputs = ((seq_len - convHeight) + 1)
        numMaxPoolOutputs = int(((numConvOutputs-maxPoolSize)/maxPoolStride)+1)
        gruHiddenVecSize = 35
        numFCNodes = 45
        numOutputNodes = 1

        # this fixes an implementation bug in Keras. If this is not true,
        # then the code runs much more slowly
        assert maxPoolSize%maxPoolStride == 0

        self.add_affinity_layers(
            tf_name, 
            input_name=name_prefix+'one_hot_sequence', 
            output_name=name_prefix+'binding_affinities',
            use_all_binding_subdomains=True)

        self.add_node(
            NormalizedOccupancy(1),
            name=name_prefix + 'occupancies',
            input=name_prefix + 'binding_affinities')

        self.add_node(
            MaxPooling2D(pool_size=(50,1)),
            name=name_prefix + 'maxed_occupancies',
            input=name_prefix + 'occupancies'
        )
        
        #self.add_node(
        #    Lambda(lambda x: K.max(K.batch_flatten(x), axis=1, keepdims=True)),
        #    name=name_prefix+'max', 
        #    input=name_prefix+'maxed_occupancies')

        
        self.add_node(
            Flatten(),
            input=name_prefix+'maxed_occupancies',
            name=name_prefix+'occupancies_flat'
        )
        self.add_node(
            Dense(256, activation='sigmoid'), # numOutputNodes
            input=name_prefix+'occupancies_flat',
            name=name_prefix+'dense1'            
        ) 
        self.add_node(
            Dropout(0.5),
            name=name_prefix + 'dense2',
            input=name_prefix + 'dense1')
        self.add_node(
            Dense(1, activation='sigmoid'), # numOutputNodes
            input=name_prefix+'dense2',
            name=name_prefix+'dense'            
        )
       
        """
        self.add_node(
            Dropout(0.5),
            name=name_prefix + 'occupancies1',
            input=name_prefix + 'occupancies')

        # MaxPooling2D(pool_size=(1,maxPoolSize), stride=(1,maxPoolStride))
        #self.model.add(Reshape((numConv,numMaxPoolOutputs)))
        #self.model.add(Permute((2,1)))
        # make the number of max pooling outputs the time dimension
        #self.model.add(GRU(output_dim=gruHiddenVecSize,return_sequences=True))
        #self.model.add(TimeDistributedDense(numFCNodes,activation="relu"))
        #self.model.add(Reshape((numFCNodes*numMaxPoolOutputs,)))
        self.add_node(
            Flatten(),
            input=name_prefix+'occupancies1',
            name=name_prefix+'occupancies_flat'
        )
        self.add_node(
            Dense(1, activation='sigmoid'), # numOutputNodes
            input=name_prefix+'occupancies_flat',
            name=name_prefix+'dense'            
        )
        """
        self.add_output(
            name=name_prefix+'output', 
            input=name_prefix+'dense')
        
        assert name_prefix + 'output' not in self.named_inputs
        self.named_inputs[name_prefix + 'output'] = labels
        self.named_losses[name_prefix + 'output'] = 'mse' #expected_F1_loss
        #expected_F1_loss # 'mse' # 'binary_crossentropy' 

    def add_chipseq_layer(self, tf_id, tf_name):
        train_start_index = int(self.num_samples*0.1)
        genome_fasta = FastaFile(
            load_genome_metadata(1).filename) # annotation_id 
        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            tf_id, # args.tf_id,
            1, #args.annotation_id,
            500, # args.half_peak_width, 
            self.num_samples, #args.max_num_peaks_per_sample, 
            include_ambiguous_peaks=False,
            order_by_accessibility=True)
        # group peaks by their sample
        sample_grpd_peaks = defaultdict(list)
        for pk in peaks_and_labels:
            sample_grpd_peaks[pk.sample].append(pk)
        peaks_seqs, peaks_labels = {}, {}
        for sample_peaks in sample_grpd_peaks.itervalues():
            shuffle(sample_peaks)
        for sample_id, sample_pks in sample_grpd_peaks.iteritems():
            print "Adding ChIP-seq data for %s (sample ID %s)" % (
                    tf_name, sample_id)
            name_prefix = 'invivo_%s_%s_' % (tf_name, sample_id) 
            fwd_seqs = encode_peaks_sequence_into_binary_array(
                [pk.peak for pk in sample_pks], genome_fasta)[:,None,:,:]
            labels = np.array([pk.label for pk in sample_pks])
            print round(float(labels.sum())/labels.shape[0], 3)
            self.add_invivo_layer(
                tf_name=tf_name, 
                sample_id=sample_id, 
                fwd_seqs=fwd_seqs[train_start_index:,:,:,:], 
                labels=labels[train_start_index:])

            name_prefix = 'invivo_%s_%s_' % (tf_name, sample_id) 
            self.named_validation_inputs[
                name_prefix+'one_hot_sequence'] = fwd_seqs[:train_start_index,:,:,:]
            self.named_validation_inputs[name_prefix+'output'] = labels[
                :train_start_index]
            #break
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
            raise ValueError, "No background sequences."

        with optional_gzip_open(fnames[max(fnames.keys())]) as fp:
            bnd_seqs = load_fastq(fp, self.num_samples/2)
        with optional_gzip_open(bg_fname) as fp:
            unbnd_seqs = load_fastq(fp, self.num_samples/2)
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
            optimizer=Adam()
        )
        return compile(*args, **kwargs)

    def fit(self, 
            data=None, 
            validation_split=0.1, 
            batch_size=100, 
            nb_epoch=100,
            *args, **kwargs):
        if data is None: data = self.named_inputs
        monitor_accuracy_cb = MonitorAccuracy()
        fit = functools.partial(
            super(JointBindingModel, self).fit,
            data=data,
            validation_split=validation_split,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[monitor_accuracy_cb,],
            shuffle=True
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

def main():
    n_samples = 10000
    model = JointBindingModel(n_samples)

    model.add_selex_layer(441) # MAX
    model.add_selex_layer(202) # YY1
    #model.add_chipseq_layer('T011266_1.02', 'MAX') 
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=5)
    return

    import psycopg2
    conn = psycopg2.connect(
        "host=%s dbname=%s user=%s" % ('mitra', 'cisbp', 'nboley'))
    cur = conn.cursor()
    query = """
      SELECT selex_experiments.tf_id, tf_name, array_agg(distinct selex_exp_id)
        FROM roadmap_matched_chipseq_peaks, selex_experiments 
       WHERE roadmap_matched_chipseq_peaks.tf_id = selex_experiments.tf_id 
    GROUP BY selex_experiments.tf_id, tf_name
    ORDER BY tf_name;
    """
    cur.execute(query)
    res = list(cur.fetchall())[:3]
    
    for tf_id, tf_name, selex_exp_ids in res:
        for selex_exp_id in selex_exp_ids:
            try: 
                model.add_selex_layer(selex_exp_id)
            except Exception, inst:
                print inst
                continue

    for tf_id, tf_name, selex_exp_ids in res:
        try:
            model.add_chipseq_layer(tf_id, tf_name)
        except Exception, inst:
            print "Couldnt load ChIP-seq data for %s" % tf_name
            continue
        else:
            break
    
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
