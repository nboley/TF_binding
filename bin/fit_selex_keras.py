import functools
from collections import defaultdict

import numpy as np

from fit_selex import SelexDBConn, load_fastq, optional_gzip_open
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
from keras.layers.core import Dense, Activation, Flatten, Merge, TimeDistributedDense

import theano
import theano.tensor as TT

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

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_motifs': self.nb_motifs,
                  'motif_len': self.motif_len, 
                  'use_three_base_encoding': self.use_three_base_encoding,
                  'init': self.init.__name__}
        base_config = super(ConvolutionDNASequenceBinding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
            
    
class ConvolutionBindingSubDomains(Layer):
    def __init__(
            self,
            nb_domains, 
            domain_len='full', 
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
        if self.domain_len == 'full':
            self.domain_len = self.input_shape[2]
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
        # # [:,:,::-1,::-1]
        rc_rv = K.conv2d(X[:,:,:,1:2], self.W, border_mode='valid') \
                + K.reshape(self.b, (1, self.nb_domains, 1, 1))
        return K.concatenate((fwd_rv, rc_rv), axis=3)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_domains': self.nb_domains,
                  'domain_len': self.domain_len, 
                  'init': self.init.__name__}
        base_config = super(ConvolutionBindingSubDomains, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
                self.input_shape[2]-self.domain_len+1,
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


class MonitorAccuracy(keras.callbacks.Callback):    
    def on_epoch_end(self, *args, **kwargs):
        train_accs = self.model.calc_accuracy(val=False)
        val_accs = self.model.calc_accuracy(val=True)
        #print "acc: %.2f - val_acc: %.2f" % (train_acc, val_acc) 
        for key in train_accs.iterkeys():
            print "%s acc: %.2f - val_acc: %.2f" % (
                key.ljust(30), train_accs[key], val_accs[key]) 

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
        super(Graph, self).__init__(*args, **kwargs)

    def add_invitro_layer(self, tf_name, fwd_seqs, shape_seqs, labels):
        num_conv = 10
        conv_size = 7

        name_prefix = 'invitro_%s_' % tf_name 

        self.add_input(
            name=name_prefix+'one_hot_sequence', 
            input_shape=(None, fwd_seqs.shape[2], 4))
        assert name_prefix+'one_hot_sequence' not in self.named_inputs
        self.named_inputs[name_prefix+'one_hot_sequence'] = fwd_seqs

        self.add_node(
            ConvolutionDNASequenceBinding(
                num_conv, # numConv
                conv_size, # convWidth, 
                #init=initial_model
            ), 
            name=name_prefix+'binding_sub_domains', 
            input=name_prefix+'one_hot_sequence'
        )

        if shape_seqs is not None:
            self.add_input(
                name=name_prefix+'shape', 
                input_shape=(None, fwd_seqs.shape[2], 12))
            assert name_prefix+'shape' not in self.named_inputs
            self.named_inputs[name_prefix+'shape'] = shape_seqs

            self.add_node(
                ConvolutionDNAShapeBinding(
                    numConv, # numConv
                    14, # convWidth, 
                    #, init=initial_model
                ), 
            name=name_prefix+'shape_binding_affinities', 
            input=name_prefix+'shape'
            )


        if shape_seqs is None:
            self.add_node(
                ConvolutionBindingSubDomains(
                    num_conv, # numConv
                    'full', # convWidth,
                ), 
                name=name_prefix + 'binding_affinities', 
                input=name_prefix + 'binding_sub_domains'
            )
        else:
            self.add_node(
                ConvolutionBindingSubDomains(
                    num_conv, # numConv
                    'full', # convWidth,
                ), 
                name=name_prefix + 'binding_affinities',
                inputs=[
                    name_prefix + 'binding_sub_domains', 
                    name_prefix + 'shape_binding_affinities'
                ],
                merge_mode='concat',
                concat_axis=1
            )

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

        self.named_losses[name_prefix + 'output'] = my_mean_squared_error # 'mse'

        return

    def add_selex_layer(self, selex_exp_id):
        # load connect to the DB, and find the factor name
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', selex_exp_id)
        factor_name = selex_db_conn.get_factor_name()

        # load the sequencess
        print "Loading sequences for %s (exp ID %i)" % (
                factor_name, selex_exp_id)
        fnames, bg_fname = selex_db_conn.get_fnames()
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
        exp_name = '%s_%i' % (factor_name, selex_exp_id) 
        self.add_invitro_layer(
            exp_name,
            fwd_seqs[train_start_index:,:,:,:], 
            None, 
            labels[train_start_index:])

        name_prefix = 'invitro_%s_' % exp_name 
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

def full_model(fwd_seqs, shape_seqs, labels):
    model = JointBindingModel()
    model.add_invitro_layer(
        'test', fwd_seqs, None, labels) # shape_seqs
    model.add_invitro_layer(
        'test2', fwd_seqs, None, labels) # shape_seqs
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=100)

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

def load_selex_seqs(experiment_id, max_num_seqs_per_file=5000):
    from fit_selex import SelexDBConn, load_fastq, optional_gzip_open
    selex_db_conn = SelexDBConn(
        'mitra', 'cisbp', 'nboley', experiment_id)
    fnames, bg_fname = selex_db_conn.get_fnames()
    with optional_gzip_open(fnames[max(fnames.keys())]) as fp:
        bnd_seqs = load_fastq(fp, max_num_seqs_per_file)
    with optional_gzip_open(bg_fname) as fp:
        unbnd_seqs = load_fastq(fp, max_num_seqs_per_file)
    factor_name = selex_db_conn.get_factor_name()
    return factor_name, get_coded_seqs_and_labels(unbnd_seqs, bnd_seqs)

def main():
    model = JointBindingModel(100000)
    for exp_id in (473, 441, 371, 83, 484, 35, 393, 30, 254, 202, 490, 91): # 441, 371, 
        model.add_selex_layer(exp_id)

    print "Compiling Model"
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=100)

    pass
    

main()
