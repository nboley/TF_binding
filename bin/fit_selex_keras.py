import numpy as np

from fit_selex import parse_arguments
from pyDNAbinding.binding_model import ConvolutionalDNABindingModel
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda, Layer
from keras import initializations
from keras import backend as K
import theano.tensor

def bad_sequential():
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    
    seq_len = 20
    conv_size = 7
    feature_size = 4 #10

    model = Sequential()
    model.add(
        ConvolutionDNABinding(2, conv_size, input_shape=(10, seq_len, feature_size))
    )
    model.add(Activation('sigmoid'))
    model.add(Flatten())
    model.add(Dense(output_dim=1))
    model.compile(loss="MSE", class_mode='binary', optimizer=Adam())

    #history = MonitorAccuracy()
    model.fit(
        fwd_seqs,
        labels,
        validation_split=0.1,
        batch_size=100,
        nb_epoch=20,
        show_accuracy=True,
        callbacks=[]
    )

def my_graph_model():
    model = Graph()
    model.add_input(name='fwd_seq', input_shape=(10, seq_len, feature_size))
    #model.add_node(
    #    ConvolutionDNABinding(2, conv_size),
    #    name='fwd_binding_site_affinities1', input='fwd_seq')

    #model.add_node(
    #    ConvolutionDNABinding(2, conv_size),
    #    name='fwd_binding_site_affinities1', input='fwd_seq')
    #model.add_node(Activation('sigmoid'), 
    #               input='fwd_binding_site_affinities1', 
    #               name='fwd_binding_site_affinities2')
    
    model.add_node(
        ConvolutionDNABinding(
            5, # numConv
            conv_size, # convWidth, 
            feature_size, # convHeight
            activation="sigmoid", 
            init="he_normal",
            input_shape=(1, seq_len, feature_size)
        ), name='fwd_binding_site_affinities1', input='fwd_seq')
    #model.add_node(Activation('sigmoid'), 
    #               input='fwd_binding_site_affinities1', 
    #               name='fwd_binding_site_affinities2')

    model.add_node(Flatten(), 
                   input='fwd_binding_site_affinities2', 
                   name='fwd_binding_site_affinities')

    model.add_node(
        Dense(output_dim=1),
        name='dense', 
        inputs=['fwd_binding_site_affinities','fwd_binding_site_affinities'])
    model.add_output(name='output', input='dense')
    model.compile(
        loss={'output': "msle"}, optimizer=Adam()
    )


def get_coded_seqs_and_labels(rnds_and_seqs, background_seqs, rnd=None):
    if rnd == None: rnd = max(rnds_and_seqs.keys())
    bnd_seqs = rnds_and_seqs[rnd]  
    num_seqs = min(len(background_seqs), len(bnd_seqs))
    unbnd_seqs = FixedLengthDNASequences(background_seqs[:num_seqs])
    bnd_seqs = FixedLengthDNASequences(bnd_seqs[:num_seqs])
    
    permutation = np.random.permutation(2*num_seqs)
    fwd_one_hot_seqs = np.vstack(
        (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs) #
    )[permutation,None,:,:]
    ## we keep this in case there is a more complicated encoding
    #rc_one_hot_seqs = np.vstack(
    #    (unbnd_seqs.rc_one_hot_coded_seqs, bnd_seqs.rc_one_hot_coded_seqs)
    #)[permutation,None,:,:]
    labels = np.hstack(
        (np.zeros(len(unbnd_seqs), dtype='float32'), 
         np.ones(len(bnd_seqs), dtype='float32'))
    )[permutation]
    return fwd_one_hot_seqs, labels

import keras.callbacks
class MonitorAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, *args, **kwargs):
        res = self.model.predict(
            {'sequence': self.model.validation_data[0]}
        )
        val_acc = (
            1.0*(res['output'].round() == self.model.validation_data[-2]).sum()
            /self.model.validation_data[-2].shape[0]
        )

        res = self.model.predict(
            {'sequence': self.model.training_data[0]}
        )
        train_acc = (
            1.0*(res['output'].round() == self.model.training_data[-2]).sum()
            /self.model.training_data[-2].shape[0]
        )

        print "acc: %.2f - val_acc: %.2f" % (train_acc, val_acc) 

class ConvolutionDNABinding(Layer):
    def __init__(
            self, nb_motifs, motif_len, base_encoding_size, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.base_encoding_size = base_encoding_size
        assert self.base_encoding_size == 4
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
        super(ConvolutionDNABinding, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.W_shape = (
            self.nb_motifs, 1, self.motif_len, self.base_encoding_size)
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
        fwd_rv = K.conv2d(X, self.W)  \
                 + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X, self.W[:,:,::-1,::-1]) \
                + K.reshape(self.b, (1, self.nb_motifs, 1, 1))
        return theano.tensor.stack((fwd_rv, rc_rv), axis=3)
        #return rc_rv

def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs,
      ofname_prefix
     ) = parse_arguments()
    fwd_seqs, labels = get_coded_seqs_and_labels(
        rnds_and_seqs, background_seqs, None)

    from keras.models import Graph
    from keras.optimizers import Adam, Adamax, RMSprop, SGD
    from keras.layers.core import Dense, Activation, Flatten
    
    seq_len = 20
    conv_size = 7
    feature_size = 4 #10

    model = Graph()
    model.add_input(name='sequence', input_shape=(1, seq_len, 2*feature_size))
    model.add_node(
        ConvolutionDNABinding(
            1, # numConv
            conv_size, # convWidth, 
            feature_size # convHeight,
            , init=initial_model
        ), name='binding_site_affinities1', input='sequence')
    model.add_node(Activation('sigmoid'), 
                   input='binding_site_affinities1', 
                   name='binding_site_affinities2')
    model.add_node(Flatten(), 
                   input='binding_site_affinities2', 
                   name='binding_site_affinities')
    model.add_node(
        Dense(output_dim=1),
        name='dense', 
        inputs=['binding_site_affinities','binding_site_affinities'])
    model.add_output(name='output', input='dense')
    model.compile(
        loss={'output': "mse"}, optimizer=Adam()
    )
    monitor_accuracy_cb = MonitorAccuracy()
    model.fit(
        {'sequence': fwd_seqs, #np.concatenate((fwd_seqs, rc_seqs), axis=3),
         'output': labels},
        validation_split=0.1,
        batch_size=100,
        nb_epoch=10,
        callbacks=[monitor_accuracy_cb,],
    )
    
    
    """
    unbnd_affinities = -unbnd_seqs.score_binding_sites(
        initial_model, 'BOTH_FLAT')
    print unbnd_affinities.min(1).mean()

    bnd_affinities = -bnd_seqs.score_binding_sites(
        initial_model, 'BOTH_FLAT')
    print bnd_affinities.min(1).mean()
    """

    """
    selex_seqs = {}
    for rnd, seqs in rnds_and_seqs.iteritems():
        selex_seqs[rnd] = FixedLengthDNASequences(seqs)
        affinities = -selex_seqs[rnd].score_binding_sites(
            initial_model, 'BOTH_FLAT')
        occs = calc_occ(-8, affinities)
        #occs = occs.reshape((occs.shape[0], occs.shape[1]*occs.shape[2]))
        print occs.shape
    print selex_seqs
    """

main()
