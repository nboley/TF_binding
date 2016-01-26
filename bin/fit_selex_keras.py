import numpy as np

from fit_selex import parse_arguments
from pyDNAbinding.binding_model import ConvolutionalDNABindingModel
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ, R, T

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda, Layer
from keras import initializations
from keras import backend as K

import theano
import theano.tensor as TT

def get_coded_seqs_and_labels(rnds_and_seqs, background_seqs, rnd=None):
    if rnd == None: rnd = max(rnds_and_seqs.keys())
    bnd_seqs = rnds_and_seqs[rnd]  
    num_seqs = min(len(background_seqs), len(bnd_seqs))
    unbnd_seqs = FixedLengthDNASequences(background_seqs[:num_seqs])
    bnd_seqs = FixedLengthDNASequences(bnd_seqs[:num_seqs])
    
    permutation = np.random.permutation(2*num_seqs)
    fwd_one_hot_seqs = np.vstack(
        (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs) # _one_hot
    )[permutation,None,:,:]
    ## we keep this in case there is a more complicated encoding
    #rc_one_hot_seqs = np.vstack(
    #    (unbnd_seqs.rc_one_hot_coded_seqs, bnd_seqs.rc_one_hot_coded_seqs) # one_hot_
    #)[permutation,None,:,:]

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
class MonitorAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, *args, **kwargs):
        res = self.model.predict(
            {'one_hot_sequence': self.model.validation_data[0],
             'shape': self.model.validation_data[1]}
        )
        val_acc = (
            1.0*(res['output'].round() == self.model.validation_data[-2]).sum()
            /self.model.validation_data[-2].shape[0]
        )

        res = self.model.predict(
            {'one_hot_sequence': self.model.training_data[0],
             'shape': self.model.training_data[1]}
        )
        train_acc = (
            1.0*(res['output'].round() == self.model.training_data[-2]).sum()
            /self.model.training_data[-2].shape[0]
        )

        print "acc: %.2f - val_acc: %.2f" % (train_acc, val_acc) 

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
            self, nb_motifs, motif_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        
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

    def build(self):
        self.W_shape = (
            self.nb_motifs, 1, self.motif_len, self.input_shape[3])
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
        fwd_rv = K.conv2d(X, 
                          self.W, 
                          border_mode='valid') \
                 + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X, 
                         self.W[:,:,::-1,::-1], 
                         border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        return K.concatenate((fwd_rv, rc_rv), axis=3)

class ConvolutionBindingSubDomains(Layer):
    def __init__(
            self, nb_domains, domain_len, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_domains = nb_domains
        self.domain_len = domain_len

        self.input = K.placeholder(ndim=4)
        
        self.init = lambda x: (
            initializations.get(init)(x), 
            K.zeros((self.nb_domains,)) 
        )
        super(ConvolutionBindingSubDomains, self).__init__(**kwargs)

    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        assert self.input_shape[3] == 2
        self.W_shape = (
            self.nb_domains, self.input_shape[1], self.domain_len, 1)
        self.W, self.b = self.init(self.W_shape)
        self.params = [self.W, self.b]
    
    @property
    def output_shape(self):
        return (# number of obseravations
                self.input_shape[0], 
                self.nb_domains,
                # sequence length minus the motif length
                self.input_shape[2]-self.domain_len+1,
                2)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        fwd_rv = K.conv2d(X[:,:,:,0:1], self.W, border_mode='valid')  \
                 + K.reshape(self.b, (1, self.nb_domains, 1, 1))
        rc_rv = K.conv2d(X[:,:,:,1:2], self.W[:,:,::-1,::-1], border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_domains, 1, 1))
        return K.concatenate((fwd_rv, rc_rv), axis=3)

class NormalizedOccupancy(Layer):
    def __init__(
            self, domain_len, **kwargs):
        self.domain_len = domain_len
        self.input = K.placeholder(ndim=4)        
        super(NormalizedOccupancy, self).__init__(**kwargs)

    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        assert self.input_shape[3] == 2
        self.chem_affinity = K.variable(-6.0)
        self.params = [self.chem_affinity]
    
    @property
    def output_shape(self):
        return (self.input_shape[0], 1, 1, 1)
        return (# number of obseravations
                self.input_shape[0], 
                self.input_shape[1],
                # sequence length minus the motif length
                self.input_shape[2], # -self.domain_len+1
                2)
    
    def get_output(self, train=False):
        X = self.get_input(train)
        #return TT.exp(theano_calc_log_occs(X, -16))
        p_op = theano.printing.Print("="*40)
        log_occs = TT.flatten(
            theano_calc_log_occs(X, self.chem_affinity), outdim=2)
        scale_factor = K.max(log_occs, axis=1)
        centered_log_occs = log_occs - K.reshape(scale_factor, (X.shape[0],1))
        norm_occs = K.sum(K.exp(centered_log_occs), axis=1)
        log_sum_log_occs = TT.reshape(K.log(norm_occs) + scale_factor, (X.shape[0],1))
        return K.exp(log_sum_log_occs)
        norm_log_occs = log_occs-log_sum_log_occs
        norm_occs = K.exp(norm_log_occs)
        return K.reshape(norm_occs, X.shape)
        return K.max(norm_occs, axis=1) #K.reshape(norm_occs, X.shape)

def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs,
      ofname_prefix
     ) = parse_arguments()
    fwd_seqs, shape_seqs, labels = get_coded_seqs_and_labels(
        rnds_and_seqs, background_seqs, None) # None
    
    from keras.models import Graph
    from keras.optimizers import Adam, Adamax, RMSprop, SGD
    from keras.layers.core import Dense, Activation, Flatten, Merge

    num_conv = 1
    seq_len = 20
    conv_size = 1

    model = Graph()
    
    model.add_input(
        name='one_hot_sequence', input_shape=(None, seq_len, 4))
    model.add_node(
        ConvolutionDNASequenceBinding(
            num_conv, # numConv
            conv_size, # convWidth, 
            #init=initial_model
        ), name='binding_sub_domains', input='one_hot_sequence')

    #model.add_node(
    #    ConvolutionBindingSubDomains(
    #        10, # numConv
    #        4, # convWidth, 
    #    ), name='domains_binding_affinities', input='binding_sub_domains')

    model.add_input(
        name='shape', input_shape=(None, seq_len, 12))
    model.add_node(
        ConvolutionDNAShapeBinding(
            num_conv, # numConv
            conv_size, # convWidth, 
            #, init=initial_model
        ), name='shape_binding_affinities', input='shape')

    model.add_node(
        ConvolutionBindingSubDomains(
            1, # numConv
            conv_size, # convWidth,
        ), 
        name='binding_affinities', 
        inputs=['binding_sub_domains', 'shape_binding_affinities'],
        merge_mode='concat',
        concat_axis=1
    )
    model.add_node(
        NormalizedOccupancy(1),
        name='occupancies',
        input='binding_affinities')

    #model.add_node(
    #    Activation(lambda x: TT.exp(theano_calc_log_occs(-x,0))), 
    #    input='binding_affinities',
    #    name='occupancies')

    model.add_node(Flatten(), 
                   input='occupancies', 
                   name='occupancies_flat')
    
    model.add_node(
        Dense(output_dim=1),
        name='dense', 
        input='occupancies_flat')
    
    
    model.add_output(name='output', input='dense')
    model.compile(
        loss={'output': "mse"}, optimizer=Adam()
    )
    monitor_accuracy_cb = MonitorAccuracy()
    model.fit(
        {'one_hot_sequence': fwd_seqs,
         'shape': shape_seqs,
         'output': labels},
        validation_split=0.1,
        batch_size=100,
        nb_epoch=100,
        callbacks=[monitor_accuracy_cb,],
        shuffle=True
    )

main()
