import numpy as np

from fit_selex import parse_arguments
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.misc import calc_occ

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda, Layer
from keras import initializations
from keras import backend as K
import theano.tensor

class ConvolutionDNABinding(Convolution2D):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, nb_row, nb_col)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_row, nb_col, nb_filter)` if dim_ordering='tf'.
    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of bases in the convolution kernel.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    def __init__(self, nb_filter, base_encoding_size=4, **kwargs):
        Convolution2D.__init__(self, nb_filter, **kwargs)

    @property
    def output_shape(self):
        base_output_shape = Convolution2D.output_shape(self)
        # double the output dimension to account for the 
        # forward and reverse complement sequence
        base_output_shape[2] *= 2

    def get_output(self, train=False):
        X = self.get_input(train)
        fwd_conv_out = K.conv2d(X, self.W, strides=self.subsample,
                                border_mode=self.border_mode,
                                dim_ordering=self.dim_ordering,
                                image_shape=self.input_shape,
                                filter_shape=self.W_shape)
        rc_conv_out = K.conv2d(X, np.flipud(np.fliplr(self.W)), 
                               strides=self.subsample,
                               border_mode=self.border_mode,
                               dim_ordering=self.dim_ordering,
                               image_shape=self.input_shape,
                               filter_shape=self.W_shape)
        conv_out = np.vstack(fwd_conv_out, rc_conv_out)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionDNABinding(Convolution2D):
    def __init__(self, nb_motifs, motif_len, base_encoding_size=4, **kwargs):
        Convolution2D.__init__(
            self, nb_motifs, motif_len, base_encoding_size, **kwargs)

    @property
    def output_shape(self):
        base_output_shape = list(super(ConvolutionDNABinding, self).output_shape)
        if self.dim_ordering == 'th':
            base_output_shape[3] *= 2
            #return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            base_output_shape[2] *= 2
            #return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Unrecognized dim_ordering: %s (This should never happen - Keras must have added a new dimension ordering)' % self.dim_ordering)
        return tuple(base_output_shape)

    def get_output(self, train=False):
        X = self.get_input(train)
        fwd_conv_out = K.conv2d(X, self.W, strides=self.subsample,
                                border_mode=self.border_mode,
                                dim_ordering=self.dim_ordering,
                                image_shape=self.input_shape,
                                filter_shape=self.W_shape)
        rc_conv_out = K.conv2d(X, self.W[:,::-1,::-1,:], 
                               strides=self.subsample,
                               border_mode=self.border_mode,
                               dim_ordering=self.dim_ordering,
                               image_shape=self.input_shape,
                               filter_shape=self.W_shape)
        conv_out = theano.tensor.concatenate((fwd_conv_out, rc_conv_out), 3)
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 3))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 2, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

def get_coded_seqs_and_labels(rnds_and_seqs, background_seqs, rnd=None):
    if rnd == None: rnd = max(rnds_and_seqs.keys())
    bnd_seqs = rnds_and_seqs[rnd]  
    num_seqs = min(len(background_seqs), len(bnd_seqs))
    unbnd_seqs = FixedLengthDNASequences(background_seqs[:num_seqs])
    bnd_seqs = FixedLengthDNASequences(bnd_seqs[:num_seqs])
    
    permutation = np.random.permutation(2*num_seqs)
    fwd_one_hot_seqs = np.vstack(
        (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs)
    )[permutation,None,:,:]
    rc_one_hot_seqs = np.vstack(
        (unbnd_seqs.rc_one_hot_coded_seqs, bnd_seqs.rc_one_hot_coded_seqs)
    )[permutation,None,:,:]
    labels = np.hstack(
        (np.zeros(len(unbnd_seqs), dtype='float32'), 
         np.ones(len(bnd_seqs), dtype='float32'))
    )[permutation]
    return fwd_one_hot_seqs, rc_one_hot_seqs, labels

import keras.callbacks
class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, *args, **kwargs):
        res = self.model.predict(
            {'fwd_seq': self.model.validation_data[0],
             'rc_seq': self.model.validation_data[1]}
        )
        val_acc = (
            1.0*(res['output'].round() == self.model.validation_data[-2]).sum()
            /self.model.validation_data[-2].shape[0]
        )

        res = self.model.predict(
            {'fwd_seq': self.model.training_data[0],
             'rc_seq': self.model.training_data[1]}
        )
        train_acc = (
            1.0*(res['output'].round() == self.model.training_data[-2]).sum()
            /self.model.training_data[-2].shape[0]
        )

        print "acc: %.2f - val_acc: %.2f" % (train_acc, val_acc) 
        #print res['output']
        #print self.model.validation_data[-2]
        #print "="*50

class ConvolutionDNABinding(Layer):
    def __init__(
            self, nb_motifs, motif_len, base_encoding_size=4, 
            init='glorot_uniform', 
            **kwargs):
        self.nb_motifs = nb_motifs
        self.motif_len = motif_len
        self.base_encoding_size = base_encoding_size
        assert self.base_encoding_size == 4
        self.input = K.placeholder(ndim=4) 
        self.init = initializations.get(init)
        super(ConvolutionDNABinding, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.W_shape = (
            self.nb_motifs, 1, self.motif_len, self.base_encoding_size)
        self.W = self.init(self.W_shape)

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
        fwd_rv = K.conv2d(X, self.W)
        rc_rv = K.conv2d(X, self.W[:,:,::-1,::-1])
        return theano.tensor.stack((fwd_rv, rc_rv), axis=3)
        #return rc_rv

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

def bad_sequential():
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    
    seq_len = 20
    conv_size = 7
    feature_size = 4 #10

    model = Sequential()
    #model.add(
    #    ConvolutionDNABinding(2, conv_size, input_shape=(10, seq_len, feature_size))
    #)
    model.add(
        Convolution2D(
            10, # numConv
            conv_size, # convWidth, 
            feature_size, # convHeight
            activation="sigmoid", 
            init="he_normal",
            input_shape=(1, seq_len, feature_size)
        ))

    #model.add(Activation('sigmoid'))
    model.add(Flatten())
    model.add(Dense(output_dim=1))
    model.compile(loss="binary_crossentropy", class_mode='binary', optimizer=Adam())


    #model.add(Permute((1,2)))

    #model.add(MaxPooling2D((seq_len-conv_size+1,1), (1,1), 'valid'))
    #model.add(Flatten())
    #model.add(Activation('sigmoid'))
    #model.add(Dense(output_dim=1))

    history = LossHistory()
    model.fit(
        fwd_seqs,
        labels,
        validation_split=0.1,
        batch_size=100,
        nb_epoch=20,
        show_accuracy=True,
        callbacks=[] # history,
    )


def main():
    ( initial_model, 
      rnds_and_seqs, background_seqs, 
      selex_db_conn, 
      partition_background_seqs,
      ofname_prefix
     ) = parse_arguments()

    fwd_seqs, rc_seqs, labels = get_coded_seqs_and_labels(
        rnds_and_seqs, background_seqs, None)

    from keras.models import Sequential, Graph
    from keras.optimizers import Adam
    from keras.layers.core import Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D
    
    seq_len = 20
    conv_size = 7
    feature_size = 4 #10

    model = Graph()

    model.add_input(name='fwd_seq', input_shape=(1, seq_len, feature_size))    
    model.add_node(
        Convolution2D(
            5, # numConv
            conv_size, # convWidth, 
            feature_size, # convHeight
            input_shape=(1, seq_len, feature_size)
        ), name='fwd_binding_site_affinities1', input='fwd_seq')
    model.add_node(Activation('sigmoid'), 
                   input='fwd_binding_site_affinities1', 
                   name='fwd_binding_site_affinities2')
    model.add_node(Flatten(), 
                   input='fwd_binding_site_affinities2', 
                   name='fwd_binding_site_affinities')

    model.add_input(name='rc_seq', input_shape=(1, seq_len, feature_size))    
    model.add_node(
        Convolution2D(
            5, # numConv
            conv_size, # convWidth, 
            feature_size, # convHeight
            input_shape=(1, seq_len, feature_size)
        ), name='rc_binding_site_affinities1', input='rc_seq')
    model.add_node(Activation('sigmoid'), 
                   input='rc_binding_site_affinities1', 
                   name='rc_binding_site_affinities2')
    model.add_node(Flatten(), 
                   input='rc_binding_site_affinities2', 
                   name='rc_binding_site_affinities')

    model.add_node(
        Dense(output_dim=1),
        name='dense', 
        inputs=['fwd_binding_site_affinities','rc_binding_site_affinities'])
    model.add_output(name='output', input='dense')
    model.compile(
        loss={'output': "msle"}, optimizer=Adam()
    )
    history = LossHistory()
    model.fit(
        {'fwd_seq': fwd_seqs, 'rc_seq': rc_seqs, 'output': labels},
        validation_split=0.1,
        batch_size=100,
        nb_epoch=20,
        callbacks=[history,],
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
