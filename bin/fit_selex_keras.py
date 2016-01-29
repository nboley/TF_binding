import numpy as np

from fit_selex import parse_arguments
from pyDNAbinding.binding_model import ConvolutionalDNABindingModel
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ, R, T
from pyDNAbinding.plot import plot_bases, pyplot

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda, Layer
from keras import initializations
from keras import backend as K

from keras.models import Graph
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.layers.core import Dense, Activation, Flatten, Merge

import theano
import theano.tensor as TT

def my_mean_squared_error(y_true, y_pred):
    #clipped_pred = K.clip(y_pred, 0.01, 0.99)
    #loss = K.sum(y_true*K.log(clipped_pred))/K.sum(y_true)
    #return loss
    return K.mean((1-y_true)*y_pred) - K.mean(y_true*y_pred)
    #return K.mean(K.square(y_pred - y_true), axis=-1)

def get_coded_seqs_and_labels(rnds_and_seqs, background_seqs, rnd=None):
    if rnd == None: 
        rnd = max(
            rnd for rnd, seqs in rnds_and_seqs.items() if len(seqs) > 500)
    
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

def fancy_theano_calc_3_base_coded_affinities(one_hot_seqs, W, b):
    # make sure the convolutional filter and biases match up
    assert W.shape[1] == 1
    assert b.shape[0] == W.shape[0]
    # this was written for theano_conv2d filter call and, since Keras
    # flips the filter, we will initially flip the filter
    W = W[:,:,::-1,::-1]
    fwd_bs_base_affinities = K.conv2d(
        one_hot_seqs, W[:,:,::-1,::-1])[:,0,:]
    rc_ddg_base_cont = (
        K.concatenate((
            W[:,:,(1,0),:], 
            K.zeros_like(W[(0,),:], dtype=theano.config.floatX)
        ), axis=2) 
        - W[:,:,2,:]
    )[:,:,:,::-1]
    rc_bs_base_affinities = (
        K.conv2d(one_hot_seqs, 
                 rc_ddg_base_cont[:,:,::-1,::-1]) + W[:,:,2,:].sum()
    )
    bs_affinities = TT.stack(
        (fwd_bs_base_affinities, rc_bs_base_affinities), axis=3)
    return K.reshape(self.b, (1, self.nb_motifs, 1, 1)) + bs_affinities

import keras.callbacks
class UpdateSampleWeights(keras.callbacks.Callback):
    def __init__(self, sample_weights, *args, **kwargs):
        self._sample_weights = sample_weights
        super(UpdateSampleWeights, self).__init__(*args, **kwargs)
    def on_epoch_end(self, *args, **kwargs):
        print self.my_weights.get_value()
        self.my_weights.set_value(self.my_weights.get_value()/2)

class MonitorAccuracy(keras.callbacks.Callback):    
    def on_epoch_end(self, *args, **kwargs):
        val_res = self.model.predict(
            {'one_hot_sequence': self.model.validation_data[0],
             'shape': self.model.validation_data[1]}
        )
        val_acc = (
            1.0*(val_res['output'].round() == self.model.validation_data[-2]).sum()
            /self.model.validation_data[-2].shape[0]
        )

        train_res = self.model.predict(
            {'one_hot_sequence': self.model.training_data[0],
             'shape': self.model.training_data[1]}
        )
        train_acc = (
            1.0*(train_res['output'].round() == self.model.training_data[-2]).sum()
            /self.model.training_data[-2].shape[0]
        )
        
        #print train_res['output']
        #print train_acc
        
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
            use_three_base_encoding=True,
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.input = K.placeholder(ndim=4)
        self.use_three_base_encoding = use_three_base_encoding
        
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
        assert self.input_shape[3] == 4, "Expecting one-hot-encoded DNA seqeunce."
        if self.use_three_base_encoding:
            self.W_shape = (
                self.nb_motifs, 1, self.motif_len, 3)
        else:
            self.W_shape = (
                self.nb_motifs, 1, self.motif_len, 4)
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
            self, domain_len, chem_affinity=0.0, **kwargs):
        self.domain_len = domain_len
        self.init_chem_affinity = chem_affinity
        self.input = K.placeholder(ndim=4)        
        super(NormalizedOccupancy, self).__init__(**kwargs)

    def build(self):
        # make sure the last input dimension has dimension exactly 2, for the 
        # fwd and rc sequence
        assert self.input_shape[3] == 2
        self.chem_affinity = K.variable(self.init_chem_affinity)
        self.params = [self.chem_affinity]
    
    @property
    def output_shape(self):
        #return (self.input_shape[0],) # 1, 1, 1)
        return (# number of obseravations
                self.input_shape[0], 
                self.input_shape[1],
                # sequence length minus the motif length
                self.input_shape[2], # -self.domain_len+1
                2)
    
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


class JointBindingModel(object):
    def __init__(self):
        self.model = Graph()
        self.inputs = {}
        self.losses = {}

    def add_invitro_layer(self, tf_name,  fwd_seqs, shape_seqs, labels):
        num_conv = 10
        seq_len = 20
        conv_size = 7

        name_prefix = 'invitro_%s_' % tf_name 

        self.model.add_input(
            name=name_prefix + 'one_hot_sequence', 
            input_shape=(None, seq_len, 4))
        assert name_prefix + 'one_hot_sequence' not in self.inputs
        self.inputs[name_prefix + 'one_hot_sequence'] = fwd_seqs

        self.model.add_node(
            ConvolutionDNASequenceBinding(
                num_conv, # numConv
                conv_size, # convWidth, 
                #init=initial_model
            ), 
            name=name_prefix + 'binding_sub_domains', 
            input=name_prefix + 'one_hot_sequence'
        )

        if shape_seqs is not None:
            self.model.add_input(
                name=name_prefix + 'shape', input_shape=(None, seq_len, 12))
            assert name_prefix + 'shape' not in self.inputs
            self.inputs[name_prefix + 'shape'] = shape_seqs

            self.model.add_node(
                ConvolutionDNAShapeBinding(
                    num_conv, # numConv
                    conv_size, # convWidth, 
                    #, init=initial_model
                ), 
            name=name_prefix + 'shape_binding_affinities', 
            input=name_prefix + 'shape'
            )

        """
        self.model.add_node(
            ConvolutionBindingSubDomains(
                1, # numConv
                4, # convWidth,
            ), 
            name=name_prefix + 'binding_affinities', 
            inputs=[name_prefix + 'binding_sub_domains', 
                    name_prefix + 'shape_binding_affinities'],
            merge_mode='concat',
            concat_axis=1
        )
        """

        if shape_seqs is None:
            self.model.add_node(
                NormalizedOccupancy(1),
                name=name_prefix + 'occupancies',
                input=name_prefix + 'binding_sub_domains')
        else:
            self.model.add_node(
                NormalizedOccupancy(1),
                name=name_prefix + 'occupancies',
                inputs=[
                    name_prefix + 'binding_sub_domains', 
                    name_prefix + 'shape_binding_affinities'
                ],
                merge_mode='concat',
                concat_axis=1
            )

        self.model.add_node(Flatten(), 
                       input=name_prefix + 'occupancies', 
                       name=name_prefix + 'occupancies_flat')
        self.model.add_node(
            Dense(output_dim=1, activation='sigmoid'),
            name=name_prefix + 'dense', 
            input=name_prefix + 'occupancies_flat')

        #self.model.add_node(
        #    Lambda(lambda x: K.max(K.batch_flatten(x), axis=1, keepdims=True)),
        #    name=name_prefix + 'max', 
        #    input='occupancies')

        self.model.add_output(name=name_prefix + 'output', 
                         input=name_prefix + 'dense')
        assert name_prefix + 'output' not in self.inputs
        self.inputs[name_prefix + 'output'] = labels

        self.losses[name_prefix + 'output'] = my_mean_squared_error

        return

    def compile(self):
        self.model.compile(loss=self.losses, optimizer=Adam())
    
    def fit(self):
        #monitor_accuracy_cb = MonitorAccuracy()
        self.model.fit(
            self.inputs,
            validation_split=0.1,
            batch_size=100,
            nb_epoch=100,
            #callbacks=[monitor_accuracy_cb,],
            shuffle=True
        )

def full_model(fwd_seqs, shape_seqs, labels):
    model = JointBindingModel()
    model.add_invitro_layer(
        'test', fwd_seqs, None, labels) # shape_seqs
    model.compile()
    model.fit()

def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs,
      ofname_prefix
     ) = parse_arguments()
    fwd_seqs, shape_seqs, labels = get_coded_seqs_and_labels(
        rnds_and_seqs, background_seqs, None) # None
    full_model(fwd_seqs, None, labels) #shape_seqs, labels)

    pass
    

main()
