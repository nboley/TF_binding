import os, sys
import time
import math

from collections import OrderedDict, defaultdict
from itertools import chain

import numpy as np

from pysam import FastaFile

from pyTFbindtools.cross_validation import ClassificationResult
 
from pyTFbindtools.DB import load_genome_metadata
from pyTFbindtools.DNABindingProteins import ChIPSeqReads

from pyDNAbinding.sequence import one_hot_encode_sequence
from pyDNAbinding.binding_model import EnergeticDNABindingModel
from pyDNAbinding.plot import plot_bases, pyplot
from pyDNAbinding.misc import R, T

VERBOSE = True
USE_SOFTPLUS_ACTIVATION = False

import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates

from keras.utils.generic_utils import Progbar

import theano
import theano.tensor as TT
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.pool import Pool
from theano.tensor.nnet import sigmoid, binary_crossentropy, softplus
from theano.tensor.extra_ops import to_one_hot

from pyTFbindtools.peaks import PartitionedSamplePeaksAndChIPSeqLabels
from pyTFbindtools.selex.invivo import SelexData

from lasagne.layers import (
    Layer, MergeLayer, InputLayer, Conv2DLayer, MaxPool2DLayer, 
    DenseLayer, FlattenLayer, ExpressionLayer, GlobalPoolLayer,
    DimshuffleLayer, DropoutLayer, ConcatLayer)
from lasagne.regularization import l1, apply_penalty, regularize_layer_params

from lasagne import init
from lasagne.utils import create_param, as_theano_expression

def zero_rows_penalty(x):
    row_sums = TT.sum(abs(x), axis=0)
    return TT.sum(TT.log(1+row_sums))/x.shape[0]

def keep_positive_penalty(x):
    return TT.sum((abs(x) - x)/2 )

def mse_skip_ambig(y_true, y_pred):
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    return TT.sqr(y_pred*non_ambig - y_true*non_ambig)*y_true.shape[0]/cnts

def cross_entropy_skip_ambig(y_true, y_pred):
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    rv = binary_crossentropy(
        TT.clip(y_pred*non_ambig, 1e-6, 1-1e-6), 
        TT.clip(y_true*non_ambig, 1e-6, 1-1e-6)
    )
    return rv*y_true.shape[0]/cnts

def expected_F1_loss(y_true, y_pred, beta=0.5):
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

    return (-1e-6 -(1+beta*beta)*precision*recall)/(
        beta*beta*precision+recall+2e-6)

def expected_F1_skip_ambig(y_true, y_pred, beta=0.5):
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    rv = expected_F1_loss(
        TT.clip(y_pred*non_ambig, 1e-6, 1-1e-6), 
        TT.clip(y_true*non_ambig, 1e-6, 1-1e-6),
        beta
    )
    return rv*y_true.shape[0]/cnts

global_loss_fn = cross_entropy_skip_ambig #mse_skip_ambig #mse_skip_ambig #expected_F1_skip_ambig

def one_hot_from_indices(t, r=None):
    """
    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.

    if r is not given, r is set to max(t) + 1
    """
    if r is None:
        r = TT.max(t) + 1

    ranges = TT.shape_padleft(TT.arange(r), t.ndim)
    return TT.eq(ranges, TT.shape_padright(t, 1))

def build_embedding_conv_filter(word_len):
    rv = []
    for i in xrange(word_len):
        rv.append([0, (4**i), 2*(4**i), 3*(4**i)])
    return np.array(rv, dtype='float32').T[None,None,::-1,::-1]

def multi_base_stacked_embedding(seqs, word_len):
    #first build the embedding filter
    transform_array = build_embedding_conv_filter(word_len)
    # for each offset
    res = []
    seq_len = seqs.shape[-1]
    num_words = (seq_len-word_len+1)//word_len
    for i in xrange(word_len):
        # build the index matrices
        trans_seqs = conv2d(
            seqs[:,:,:,i:i+num_words*word_len],
            transform_array,
            border_mode='valid',
            subsample=(1,word_len)
        )
        # reconvert to one-hot
        res.append(one_hot_from_indices(trans_seqs[:,0,:,:], 4**word_len))
    return TT.concatenate(res, axis=3).dimshuffle(0,1,3,2)

def multi_base_interleaved_embedding(seqs, word_len):
    #first build the embedding filter
    transform_array = build_embedding_conv_filter(word_len)
    # build the index matrices
    trans_seqs = conv2d(
        seqs,
        transform_array,
        border_mode='valid'
    )
    rv = one_hot_from_indices(trans_seqs[:,0,:,:], 4**word_len)
    return rv.dimshuffle(0,1,3,2)

class ConvolutionDNASequenceBinding(Layer):
    def __init__(self, 
                 input,
                 nb_motifs, 
                 motif_len, 
                 use_three_base_encoding=False,
                 word_size=1, # the number of bases to group into a word
                 use_interleaved_words=False,
                 W=init.HeNormal(), 
                 b=init.Constant(-3.0),
                 **kwargs):
        super(ConvolutionDNASequenceBinding, self).__init__(input, **kwargs)

        assert use_three_base_encoding is False
        self.use_three_base_encoding = use_three_base_encoding
        
        # make sure the motif_len is divisble byt he word size
        self.motif_len = word_size*(motif_len//word_size)
        self.nb_motifs = nb_motifs
        self.word_size = word_size
        self.use_interleaved_words = (use_interleaved_words or (word_size == 1))
        
        base_size = 4**word_size
        if self.use_three_base_encoding:
            base_size -= 1
        if not self.use_interleaved_words:
            filter_shape = (
                nb_motifs, 1, base_size, self.motif_len//self.word_size)
        else:
            filter_shape = (
                nb_motifs, 1, base_size, self.motif_len)
        self.W = self.add_param(W, filter_shape, name='W')
        if use_three_base_encoding:
            self.b = self.add_param(
                b, (nb_motifs, ), name='b')
        else:
            self.b = None

    def embed_sequence(self, input):
        if self.word_size == 1:
            return input
        assert self.word_size > 1
        if self.use_interleaved_words:
            return multi_base_interleaved_embedding(
                input, self.word_size).astype('float32')
        else:
            return multi_base_stacked_embedding(
                input, self.word_size).astype('float32')

    def get_output_shape_for(self, input_shape):
        return (# number of obseravations
                self.input_shape[0],
                2,
                self.nb_motifs,
                # sequence length minus the motif length
                self.input_shape[3],
                )

    def get_single_strand_output_for(self, X):
        # embed the sequences
        X = self.embed_sequence(X)
        # actually perform the convolution
        rv = conv2d(X, 
                    self.W, 
                    border_mode='valid', 
                    subsample=(4**self.word_size,1)
        ) 
        # add the bias term
        if self.b is not None:
            rv += TT.reshape(self.b, (1, self.nb_motifs, 1, 1))
        
        # interleave the binding site affinities
        if not self.use_interleaved_words:
            rv = rv.dimshuffle(0,1,3,2)
            rv = rv.reshape((rv.shape[0],rv.shape[1],rv.shape[2]*rv.shape[3],1))
            rv = rv.dimshuffle(0,1,3,2)
        
        return rv

    def flip_sequence_get_output_for(self, input, **kwargs):
        # pad so that the output is the correct dimension
        base_pad_shape = list(input.shape[:-1])
        left_pad_shape = tuple(base_pad_shape+[int(math.ceil(self.motif_len/2))-1,])
        right_pad_shape = tuple(base_pad_shape+[int(math.floor(self.motif_len/2)  ),])
        
        # embed the sequences
        X_fwd = TT.concatenate([
            TT.zeros(left_pad_shape), input, TT.zeros(right_pad_shape)], axis=3)
        fwd_rv = self.get_single_strand_output_for(X_fwd)

        # take the reverse complement for the reverse complement section
        X_rc = TT.concatenate([
            TT.zeros(left_pad_shape), input, TT.zeros(right_pad_shape)], axis=3)
        X_rc = X_rc[:,:,::-1,::-1]
        rc_rv = self.get_single_strand_output_for(X_rc)
        rc_rv = rc_rv[:,:,:,::-1]

        # add the strands together, and reshape
        rv = TT.concatenate((fwd_rv, rc_rv), axis=2)
        rv = rv.dimshuffle((0,2,1,3))

        # no binding site should have a negative affinity to any strand of DNA
        if USE_SOFTPLUS_ACTIVATION:
            rv = softplus(rv)
        return rv
    
    def flip_filter_get_output_for(self, input, **kwargs):
        if self.use_three_base_encoding:
            X_fwd = input[:,:,1:,:]
            X_rc = input[:,:,:3,:]
        else:
            X_fwd = input
            X_rc = input
        # pad so that the output is the correct dimension
        base_pad_shape = list(X_fwd.shape[:-1])
        left_pad_shape = tuple(base_pad_shape+[int(math.ceil(self.motif_len/2))-1,])
        right_pad_shape = tuple(base_pad_shape+[int(math.floor(self.motif_len/2)),])

        X_fwd = TT.concatenate([
            TT.zeros(left_pad_shape), X_fwd, TT.zeros(right_pad_shape)], axis=3)
        X_rc = TT.concatenate([
            TT.zeros(left_pad_shape), X_rc, TT.zeros(right_pad_shape)], axis=3)
        fwd_rv = conv2d(X_fwd, self.W, border_mode='valid') 
        rc_rv = conv2d(X_rc, self.W[:,:,::-1,::-1], border_mode='valid')
        if self.b is not None:
            fwd_rv += TT.reshape(self.b, (1, self.nb_motifs, 1, 1))
            rc_rv += TT.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rv = TT.concatenate((fwd_rv, rc_rv), axis=2)
        rv = rv.dimshuffle((0,2,1,3))
        # no binding site should have a negative affinity to any strand of DNA
        if USE_SOFTPLUS_ACTIVATION:
            rv = softplus(rv)
        return rv

    def get_output_for(self, *args, **kwargs):
        # if the word size is 1 then it is faster to flip the filter than 
        # the input sequence
        if self.word_size == 1:
            return self.flip_filter_get_output_for(*args, **kwargs)
        else:
            return self.flip_sequence_get_output_for(*args, **kwargs)

class StackStrands(Layer):
    """Stack strand independent convolutions.

    This is simply checking that the size of the second dimension is two
    (because there are two strands) and then concatanating them in the 
    third dimension.
    """
    def get_output_shape_for(self, input_shape):
        # make sure there are exactly 2 signal tracks (one for each strand
        # in dimension index 1)
        assert self.input_shape[1] == 2
        return (self.input_shape[0], 
                1,
                2*self.input_shape[2],
                self.input_shape[3])
    
    def get_output_for(self, X, **kwargs):
        assert self.input_shape[1] == 2
        return X.reshape((X.shape[0], 1, 2*X.shape[2], X.shape[3]))

class AnyBoundOcc(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

    def get_output_for(self, input, **kwargs):
        input = TT.clip(input, 1e-6, 1-1e-6)
        log_none_bnd = TT.sum(
            TT.log(1-input), axis=3, keepdims=True)
        at_least_1_bnd = 1-TT.exp(log_none_bnd)
        return at_least_1_bnd

class OccupancyLayer(Layer):
    def __init__(
            self, 
            input,
            init_chem_affinity=0.0,
            dnase_signal=None,
            nsamples_and_sample_ids=None,
            **kwargs):
        super(OccupancyLayer, self).__init__(input, **kwargs)
        self.dnase_signal = dnase_signal
        if dnase_signal is not None:
            self.dnase_weight = self.add_param(
                init.Constant(1e-1),
                (self.input_shape[2],), 
                name='dnase_weight'
            )

        if nsamples_and_sample_ids is None:
            n_samples = 1
            self.sample_ids = TT.ones((1,1), dtype='float32')
        else:
            n_samples, self.sample_ids = nsamples_and_sample_ids
        
        self.chem_affinity = self.add_param(
            init.Constant(init_chem_affinity),
            (n_samples,self.input_shape[2]), 
            name='chem_affinity'
        )

    def get_output_shape_for(self, input_shape):
        return self.input_shape

    def get_output_for(self, input, **kwargs):
        sample_chem_affinities = TT.dot(self.sample_ids, self.chem_affinity)
        rv = input + sample_chem_affinities[:,None,:,None]
        if self.dnase_signal is not None:
            rv += self.dnase_weight[None, None, :, None]*self.dnase_signal[:,None,None,None]
        return sigmoid(rv)

class OccMaxPool(Layer):
    def __init__(
            self, input, num_tracks, num_bases, base_stride=None, **kwargs):
        self.num_tracks = num_tracks
        self.num_bases = num_bases
        self.base_stride = base_stride
        # make the base stride default to the number of bases int he convolution
        if self.base_stride is None:
            self.base_stride = self.num_bases
        super(OccMaxPool, self).__init__(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        assert input_shape[1] == 1, input_shape
        num_tracks = input_shape[2] if self.num_tracks == 'full' else self.num_tracks
        num_bases = input_shape[3] if self.num_bases == 'full' else self.num_bases
        base_stride = input_shape[3] if self.base_stride == 'full' else self.base_stride
        
        ds = (num_tracks, num_bases)
        st = (num_tracks, base_stride)
        return Pool.out_shape(self.input_shape, ds, st=st, ignore_border=True)

    def get_output_for(self, input, **kwargs):
        assert self.input_shape[1] == 1
        num_tracks = (
            self.input_shape[2] if self.num_tracks == 'full' 
            else self.num_tracks
        )
        num_bases = (
            self.input_shape[3] if self.num_bases == 'full' 
            else self.num_bases
        )
        base_stride = (
            self.input_shape[3] if self.base_stride == 'full' 
            else self.base_stride
        )
        ds = (num_tracks, num_bases)
        st = (num_tracks, base_stride)
        assert base_stride <= num_bases
        X = input
        rv = max_pool_2d(
            X, ds=ds, st=st, ignore_border=True, mode='max')
        return rv

class ConvolveDNASELayer(MergeLayer):
    def __init__(self, incoming, dnase, **kwargs):
        super(ConvolveDNASELayer, self).__init__([incoming, dnase], **kwargs)
        self.n_tracks = self.input_shapes[0][2]
        self.dnase_weights = self.add_param(
            init.Constant(0.0),
            (self.n_tracks,), 
            name='dnase_weights'
        )
        self.incoming = incoming
        self.dnase = dnase

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        affinities, dnase = inputs
        norm_access = TT.clip(
            dnase/TT.max(dnase, keepdims=True), 
            1e-8, 
            1.0
        )
        log_norm_access = TT.log(norm_access)
        track_access = log_norm_access.repeat(self.n_tracks, axis=2)
        return affinities + self.dnase_weights[None,:,None]*track_access

class JointBindingModel():    
    def _init_shared_affinity_params(self, use_three_base_encoding):
        # initialize the full subdomain convolutional filter
        self.num_invivo_convs = 0
        self.num_tf_specific_invitro_affinity_convs = 1
        self.num_tf_specific_invivo_affinity_convs = 7 # HERE 
        self.num_tf_specific_convs = (
            self.num_tf_specific_invitro_affinity_convs
            + self.num_tf_specific_invivo_affinity_convs
        )
        self.num_affinity_convs = (
            len(self.factor_names)*self.num_tf_specific_convs \
            + self.num_invivo_convs
        )
        self.affinity_conv_size = 32
        affinity_conv_filter_shape = (
            self.num_affinity_convs,
            1, 
            3 if use_three_base_encoding else 4, 
            self.affinity_conv_size)
        self.affinity_conv_filter = create_param(
            lambda x: init.HeNormal().sample(x),
            affinity_conv_filter_shape,
            'affinity_convolutions'
        )
        self.affinity_conv_bias = create_param(
            lambda x: init.Constant(0.0).sample(x),
            (affinity_conv_filter_shape[0],),
            'affinity_convolutions'
        )
        
        self.invitro_slices = OrderedDict()
        self.invivo_slices = OrderedDict()
        for factor_index, factor in enumerate(self.factor_names):
            self.invitro_slices[factor] = slice(
                factor_index*self.num_tf_specific_convs,
                factor_index*self.num_tf_specific_convs 
                + self.num_tf_specific_invitro_affinity_convs
            )
            self.invivo_slices[factor] = slice(
                factor_index*self.num_tf_specific_convs,
                factor_index*self.num_tf_specific_convs 
                + self.num_tf_specific_convs
            )
        
        return

    def get_tf_specific_affinity_params(self, factor_name, only_invitro):
        W = self.affinity_conv_filter[self.invitro_slices[factor_name],:,:,:]
        b = self.affinity_conv_bias[self.invitro_slices[factor_name]]
        return W, b

    def add_selex_experiment(self, selex_experiment, weight=1.0):
        seq_length = selex_experiment.fwd_seqs.shape[3]
        name = selex_experiment.id
        if VERBOSE: print "Adding", name
        
        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.labels')
        self._target_vars[name + '.labels'] = target_var
        
        network = InputLayer(
            shape=(None, 1, 4, seq_length), input_var=input_var)
        
        W, b = self.get_tf_specific_affinity_params(
            selex_experiment.factor_name, only_invitro=True)
        network = ConvolutionDNASequenceBinding(
            network, 
            nb_motifs=self.num_tf_specific_invitro_affinity_convs, 
            motif_len=self.affinity_conv_size,
            use_three_base_encoding=self._use_three_base_encoding)
            #W=W, 
            #b=b)
        network = StackStrands(network)
        network = OccMaxPool(network, 'full', 'full' )
        #network = DenseLayer(
        #    network, 
        #    1, 
        #    nonlinearity=lasagne.nonlinearities.sigmoid)

        network = OccupancyLayer(network, -2.0)
        network = FlattenLayer(network)
        
        self._networks[name + ".labels"] = network
        self._data_iterators[name] = selex_experiment.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction))
        self._losses[name] = loss #self._selex_penalty*loss

    def add_selex_experiments(self, selex_experiments):
        for exp in selex_experiments:
            self.add_selex_experiment(exp)
    
    def add_simple_chipseq_model(self, pks_and_labels):
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_ids
        name = 'invivo_%s_sequence' % "-".join(pks_and_labels.sample_ids) 

        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.labels')
        self._target_vars[name + '.labels'] = target_var
        self._multitask_labels[name + '.labels'] = pks_and_labels.factor_names
        
        network = InputLayer(
            shape=(None, 1, 4, pks_and_labels.seq_length), input_var=input_var)

        network = ConvolutionDNASequenceBinding(
            network, 
            nb_motifs=self.num_affinity_convs, 
            motif_len=self.affinity_conv_size, 
            use_three_base_encoding=self._use_three_base_encoding,
            W=self.affinity_conv_filter, 
            b=self.affinity_conv_bias)
        network = StackStrands(network)
        network = OccMaxPool(network, 2*self.num_tf_specific_convs, 16)
        network = OccupancyLayer(network, -6.0)
        self._occupancy_layers[name + ".occupancy"] = network
        network = AnyBoundOcc(network)
        network = OccMaxPool(network, 1, 'full')
        network = FlattenLayer(network)

        self._networks[name + ".labels"] = network

        prediction = lasagne.layers.get_output(network) 
        loss = TT.mean(global_loss_fn(target_var, prediction))
        self._data_iterators[name] = pks_and_labels.iter_batches
        self._losses[name] = loss

        return

    def _add_chipseq_regularization(self, affinities, target_var):
        network = OccupancyLayer(affinities, -8.0)
        network = OccMaxPool(network, 2*self.num_tf_specific_convs, 16)
        network = AnyBoundOcc(network)
        network = OccMaxPool(network, 1, 'full')
        network = FlattenLayer(network)
        self._occupancy_layers[str(target_var) + ".chipseqreg.occupancy"] = network
        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction))
        self._losses[str(target_var) + ".chipseqreg"] = (
            self._chipseq_regularization_penalty*loss)

    def add_chipseq_samples(self, pks_and_labels, include_dnase=True): 
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_ids
        name = 'invivo_%s_sequence' % "-".join(pks_and_labels.sample_ids) 
        
        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.labels')
        self._target_vars[name + '.labels'] = target_var
        self._multitask_labels[name + '.labels'] = pks_and_labels.factor_names

        network = InputLayer(
            shape=(None, 1, 4, pks_and_labels.seq_length), input_var=input_var)

        network = ConvolutionDNASequenceBinding(
            network, 
            nb_motifs=self.num_affinity_convs, 
            motif_len=self.affinity_conv_size, 
            use_three_base_encoding=self._use_three_base_encoding,
            W=self.affinity_conv_filter, 
            b=self.affinity_conv_bias)
        single_tf_affinities = StackStrands(network)

        # make sure that the single tf affinities are predictive of 
        # binding in the region
        #self._add_chipseq_regularization(single_tf_affinities, target_var)

        if include_dnase is True:
            access_input_var = TT.tensor4(name + '.dnase_cov')
            self._input_vars[name + '.dnase_cov'] = access_input_var
            dnase = InputLayer(
                shape=(None, 1, 1, pks_and_labels.seq_length), 
                input_var=access_input_var
            )
            network = ConvolveDNASELayer(single_tf_affinities, dnase)

        network = OccMaxPool(network, 1, 32, 4)
        network = Conv2DLayer(
            network, 
            len(self.invivo_factor_names),
            (2*self.num_affinity_convs,16),
            b=None,
            nonlinearity=(softplus if USE_SOFTPLUS_ACTIVATION 
                          else lasagne.nonlinearities.identity),
        )
        cobinding_penalty = 1e-5*regularize_layer_params(network,l1)
        network = DimshuffleLayer(network, (0,2,1,3))
        if include_dnase is True:
            grpd_dnase = OccMaxPool(dnase, 1, 92, 4) # try average pool XXX
            network = ConvolveDNASELayer(network, grpd_dnase)
        
        sample_ids_var = TT.matrix(name + '.sample_ids')
        self._input_vars[name + '.sample_ids'] = sample_ids_var

        network = OccupancyLayer(
            network, 
            init_chem_affinity=-10,
            nsamples_and_sample_ids=(len(self.sample_ids), sample_ids_var),
            dnase_signal=TT.log(1+TT.max(access_input_var, axis=-1)).flatten()
        )
        self._occupancy_layers[name + ".occupancy"] = network
        network = OccMaxPool(network, 1, 4)
        
        network = AnyBoundOcc(network)
        ## Test if the any bound occupancy is actually helping - if the window 
        ## sizes are small enough a maxpool may be just as good
        #network = OccMaxPool(network, 2*self.num_tf_specific_convs, 'full')
        
        network = FlattenLayer(network)
        self._networks[name + ".labels"] = network
        self._data_iterators[name] = pks_and_labels.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction)) + cobinding_penalty
        self._losses[name] = loss
        return


    def add_DIGN_chipseq_samples(self, pks_and_labels, include_DNASE=True): 
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_ids
        name = 'invivo_%s_DIGN_sequence' % "-".join(pks_and_labels.sample_ids) 
        
        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.labels')
        self._target_vars[name + '.labels'] = target_var
        self._multitask_labels[name + '.labels'] = pks_and_labels.factor_names

        n_convs = int(25*(1+np.log2(len(self.factor_names))))
        
        network = InputLayer(
            shape=(None, 1, 4, pks_and_labels.seq_length), input_var=input_var)

        if include_DNASE:
            access_input_var = TT.tensor4(name + '.dnase_cov')
            self._input_vars[name + '.dnase_cov'] = access_input_var
            access = InputLayer(
                shape=(None, 1, 1, pks_and_labels.seq_length),
                input_var=access_input_var
            )
            #access = ExpressionLayer(access, lambda x: TT.log(
            #    1e-12+x/TT.max(x, keepdims=True)))
            access = ExpressionLayer(
                access, 
                lambda x: (
                    x - TT.mean(x, axis=-1, keepdims=True)
                )/TT.std(x, axis=-1, keepdims=True)
            )
            network = ConcatLayer([access, network], axis=2)
            network = Conv2DLayer(
                network, n_convs, (5,15),
                W=lasagne.init.HeNormal(),
                nonlinearity=lasagne.nonlinearities.rectify)
        else:
            network = Conv2DLayer(
                network, n_convs, (4,15),
                nonlinearity=lasagne.nonlinearities.rectify)

        network = DropoutLayer(network, 0.2)
        network = Conv2DLayer(
            network, n_convs, (1,15),
            W=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = DropoutLayer(network, 0.2)
        network = Conv2DLayer(
            network, n_convs, (1,15),
            W=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = DropoutLayer(network, 0.2)
        network = DimshuffleLayer(network, (0,2,1,3))
        network = MaxPool2DLayer(network, (1, 35))
        network = DenseLayer(
            network, 
            len(pks_and_labels.factor_names), 
            nonlinearity=lasagne.nonlinearities.sigmoid)

        self._networks[name + ".labels"] = network
        self._data_iterators[name] = pks_and_labels.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction))
        self._losses[name] = loss
        return

    def set_all_params(self, values):
        lasagne.layers.set_all_param_values(self._networks.values(), values)

    def _build(self):
        # build the predictions dictionary
        losses = TT.stack(self._losses.values())
        regularization_loss = (0.0
            + 0.00001*apply_penalty(self.affinity_conv_filter, l1)
            + 0.0001*apply_penalty(self.affinity_conv_filter, zero_rows_penalty)
            #+ 0.01*apply_penalty(self.affinity_conv_filter, keep_positive_penalty)
        )
        ## Don't use regularization
        #regularization_loss = 0        
        params = lasagne.layers.get_all_params(
            self._networks.values(), trainable=True)
        updates = lasagne.updates.adam(
            TT.sum(losses) + regularization_loss, 
            params
        )
        #updates = lasagne.updates.apply_momentum(updates, params, 0.5)
        # build the training function
        all_vars = self._input_vars.values() + self._target_vars.values()
        self.train_fn = theano.function(
            all_vars, 
            losses + regularization_loss, 
            updates=updates, 
            on_unused_input='ignore')
        self.test_fn = theano.function(
            all_vars, losses, on_unused_input='ignore')
        
        # build the prediction functions
        self.predict_fns = {}
        for key in self._networks:
            predict_fn = theano.function(
                self._input_vars.values(), 
                lasagne.layers.get_output(
                    self._networks[key], deterministic=True),
                on_unused_input='ignore'
            )
            self.predict_fns[key] = predict_fn
        

    def __init__(self, 
                 n_samples, 
                 invivo_factor_names, 
                 sample_ids, 
                 validation_sample_ids=None,
                 invitro_factor_names=[],
                 use_three_base_encoding=False):
        self.factor_names = invivo_factor_names + [
            x for x in invitro_factor_names if x not in invivo_factor_names]
        self.invivo_factor_names = invivo_factor_names
        self.invitro_factor_names = invitro_factor_names
        
        # make sure the factor names are unique
        assert len(self.factor_names) == len(set(self.factor_names))
        self.sample_ids = sample_ids
        self._use_three_base_encoding = use_three_base_encoding
        
        self._multitask_labels = {}
        
        self._losses = OrderedDict()
        self.regularizers = OrderedDict()
        
        self._input_vars = OrderedDict()
        self._target_vars = OrderedDict()
        self._networks = OrderedDict()
        self._data_iterators = OrderedDict()
        self._occupancies_fns = None
        self._occupancy_layers = OrderedDict()
        self._deep_lifft_fns = None
        
        self.validation_losses = []
        self.validation_results = []
        self.model_params = []
        
        self._init_shared_affinity_params(self._use_three_base_encoding)

        init_sp = 2.0*len(invivo_factor_names)/(len(invitro_factor_names)+1e-6)
        self._selex_penalty = create_param(
            lambda x: init_sp, (), 'selex_penalty')
        selex_experiments = SelexData(n_samples)
        for factor_name in invitro_factor_names:
            selex_experiments.add_all_selex_experiments_for_factor(
                factor_name)
        self.add_selex_experiments(selex_experiments)

        if len(invivo_factor_names) > 0:
            self._chipseq_regularization_penalty = create_param(
                lambda x: 2.0, (), 'chipseq_penalty')
            pks = PartitionedSamplePeaksAndChIPSeqLabels(
                sample_ids, 
                factor_names=invivo_factor_names, 
                max_n_samples=n_samples, 
                validation_sample_ids=validation_sample_ids)
            #self.add_DIGN_chipseq_samples(pks)
            self.add_chipseq_samples(pks)
            #self.add_simple_chipseq_model(pks)

        self._build()
    
    def iter_data(self, *args, **kwargs_args):
        iterators = OrderedDict()
        for key, iter_inst in self._data_iterators.iteritems():
            iterators[key] = iter_inst(*args, **kwargs_args)
        assert len(iterators) > 0, 'No data iterators provided'
        
        def iter_data():
            while True:
                merged_output = {}
                for name_prefix, iterator in iterators.iteritems():
                    data_dict = next(iterator)
                    for key, data in data_dict.iteritems():
                        assert name_prefix + "." + key not in merged_output
                        merged_output[name_prefix + "." + key] = data
                ordered_merged_output = OrderedDict()
                for key in chain(
                        self._input_vars.iterkeys(), 
                        self._target_vars.iterkeys()):
                    ordered_merged_output[key] = merged_output[key]
                for key in merged_output.iterkeys():
                    if key not in ordered_merged_output:
                        ordered_merged_output[key] = merged_output[key]
                yield ordered_merged_output
        
        return iter_data()
        #return iter_weighted_batch_samples(self, iter_data(), num_oversamples)

    def get_all_params(self):
        return lasagne.layers.get_all_params(
            self._networks.values(), trainable=True)

    def _build_deeplifft_fns(self):
        real_loss_key = [
            key for key in self._losses.keys()
            if key.startswith('invivo') and key.endswith('sequence')
        ]
        assert len(real_loss_key) == 1
        real_loss_key = real_loss_key[0]
        real_loss = self._losses[real_loss_key]

        deep_lifft_fns = OrderedDict()
        for key, val in self._input_vars.iteritems():
            gradient = theano.gradient.grad(
                real_loss, val, disconnected_inputs='ignore')
            deep_lifft_fn = theano.function(
                self._input_vars.values() + self._target_vars.values(), 
                gradient,
                on_unused_input='ignore'
            )
            deep_lifft_fns[key] = deep_lifft_fn
        self._deep_lifft_fns = deep_lifft_fns

    def predict_deeplifft_scores_on_batch(self, data):
        if self._deep_lifft_fns is None:
            self._build_deeplifft_fns()
        predictions = {}
        for output_key, fn in self._deep_lifft_fns.iteritems():
            # run the occupancies prediction function, and fitler out
            # the worthless second diumnensions
            res = fn(
                *([data[key] for key in self._input_vars.keys()]
                  + [data[key] for key in self._target_vars.keys()])
            )
            predictions[output_key] = res
        return predictions
    
    def _build_predict_occupancies_fns(self):
        self._occupancies_fns = OrderedDict()
        for network_id, network in self._occupancy_layers.iteritems():
            inputs = []
            occupancy_layers = []
            for res in lasagne.layers.get_all_layers(network):
                if isinstance(res, InputLayer):
                    inputs.append(res)
                if isinstance(res, OccupancyLayer):
                    occupancy_layers.append(res)
            # For now we only deal with graphs with a single occupancy output
            assert len(occupancy_layers) == 1 
            predict_fn = theano.function(
                self._input_vars.values(),
                lasagne.layers.get_output(
                    occupancy_layers[0], deterministic=True),
                on_unused_input='ignore'
            )
            self._occupancies_fns[network_id] = predict_fn
        return

    def predict_occupancies_on_batch(self, data):
        if self._occupancies_fns is None:
            self._build_predict_occupancies_fns()
        predictions = {}
        for output_key, fn in self._occupancies_fns.iteritems():
            # runt he occupancies prediction fuinction, and fitler out
            # the worthless second diumnensions
            res = fn(
                *[data[input_key] for input_key in self._input_vars.keys()]
            )
            assert res.shape[1] == 1
            predictions[output_key] = res[:,0,:,:]
        
        return predictions
    
    def predict_on_batch(self, data):
        predictions = {}
        for output_key, fn in self.predict_fns.iteritems():
            predictions[output_key] = fn(
                *[data[input_key] for input_key in self._input_vars.keys()])
        return predictions
              
    def predict(self, batch_size):
        data_iterator = self.iter_data(
            batch_size, 
            'validation', 
            repeat_forever=False,
            balanced=False,
            shuffled=False,
        )
        pred_prbs = defaultdict(list)
        labels = defaultdict(list)
        sample_ids = defaultdict(list)
        for i, data in enumerate(data_iterator):
            for key, prbs in self.predict_on_batch(data).iteritems():
                pred_prbs[key].append(prbs)
                labels[key].append(data[key])
            try: 
                sample_ids[key].append(data[key.split(".")[0] + '.sample_ids'])
            except KeyError:
                sample_ids[key].append(
                    np.ones((data.values()[0].shape[0], 1), dtype='float32')
                )
        
        # stack everything back together
        for key in pred_prbs.iterkeys():
            pred_prbs[key] = np.vstack(pred_prbs[key])
            labels[key] = np.vstack(labels[key])
            sample_ids[key] = np.concatenate(sample_ids[key], axis=0)
        
        return pred_prbs, labels, sample_ids

    def plot_binding_models(self, prefix):
        # build the pwm
        for i in xrange(self.affinity_conv_bias.get_value().shape[0]):
            ddg_array = self.affinity_conv_filter.get_value()[i,0,:,:]
            if self._use_three_base_encoding:
                ddg_array = np.vstack(
                    (np.zeros((1, ddg_array.shape[1])), ddg_array))
            ref_energy = self.affinity_conv_bias.get_value()[i]
            mo = EnergeticDNABindingModel(ref_energy, -ddg_array.T)
            mo.build_pwm_model(-6).plot("%s-%i-pwm-fwd.png" % (prefix, i))
            mo = EnergeticDNABindingModel(ref_energy, -ddg_array[::-1,::-1].T)
            mo.build_pwm_model(-6).plot("%s-%i-pwm-rc.png" % (prefix, i))
            pyplot.close("all")
        
        if not self._use_three_base_encoding:
            for i in xrange(self.affinity_conv_bias.get_value().shape[0]):
                weights = self.affinity_conv_filter.get_value()[i,0,:,:]
                weights += self.affinity_conv_bias.get_value()[i]/len(weights.ravel())

                plot_bases(weights.T)
                pyplot.savefig("%s-%i-raw-fwd.png" % (prefix, i))
                plot_bases(weights[::-1,::-1].T)
                pyplot.savefig("%s-%i-raw-rc.png" % (prefix, i))
                pyplot.close("all")

    def predict_occupancies(self, batch_size):
        input_data =  next(self.iter_data(
            batch_size, 
            'train', 
            repeat_forever=False, 
            balanced=False, 
            shuffled=False, 
            include_chipseq_signal=True))
        res = self.predict_occupancies_on_batch(input_data)
        pred_occs = [
            data for key, data in res.iteritems() 
            if key.startswith("invivo")
        ]
        assert len(pred_occs) == 1
        obs_chipseq_signal = [
            data for key, data in input_data.iteritems() 
            if key.startswith('invivo') and key.endswith('chipseq_cov')
        ]
        dnase_signal = [
            data for key, data in input_data.iteritems() 
            if key.startswith('invivo') and key.endswith('dnase_cov')
        ]
        obs_labels = [
            data for key, data in input_data.iteritems() 
            if key.startswith('invivo') and key.endswith('output')
        ]
        assert len(obs_labels) == 1
        pred_labels = [
            data for key, data in self.predict_on_batch(input_data).iteritems()
            if key.startswith('invivo') and key.endswith('output')
        ]
        assert len(pred_labels) == 1
        assert len(obs_chipseq_signal) == 1
        assert len(dnase_signal) == 1
        return { 'obs_labels': obs_labels[0], 
                 'pred_labels': pred_labels[0], 
                 'pred_occs': pred_occs[0], 
                 'chipseq_signal': obs_chipseq_signal[0], 
                 'dnase_signal': dnase_signal[0] }

    def save(self, fname, save_train_data=False):
        """Save all of the model data.

        """
        print "Saving model and all data"
        import h5py
        with h5py.File(fname, "w") as f:
            print "Saving validation results"
            f.create_dataset('validation_losses', data=np.array(self.validation_losses))

            grp = f.create_group("validation_results")
            result_keys = None
            for epoch, results in enumerate(self.validation_results):
                epoch_grp = grp.create_group(str(epoch))
                task_grpd_results = defaultdict(dict)
                for (task, factor), result in results.iteritems():
                    task_grpd_results[task][factor] = result
                for task, task_results in task_grpd_results.iteritems():
                    task_grp = epoch_grp.create_group(str(task))
                    for factor, result in task_results.iteritems():
                        keys, vals = zip(*list(result.iter_numerical_results()))
                        if result_keys is None:
                            result_keys = keys
                        else:
                            assert result_keys == keys
                        task_grp.create_dataset(str(factor), data=np.array(vals))
            grp.attrs['result_types'] = result_keys
            
            print "Saving affinities"
            # add the affinities
            affinities_grp = f.create_group("affinities")
            affinity_conv_filters = self.affinity_conv_filter.get_value()[:,0,:,:]
            dset = affinities_grp.create_dataset(
                'conv_filters', data=affinity_conv_filters)
            affinity_ref_energies = self.affinity_conv_bias.get_value()
            dset = affinities_grp.create_dataset(
                'ref_energies', data=affinity_ref_energies)

            print "Saving parameters"
            # add all of the layer parameter values
            parameters_grp = f.create_group("parameters")
            for epoch, params in enumerate(self.model_params):
                epoch_parameters_grp = parameters_grp.create_group(str(epoch))
                for (i, name), param in params.iteritems():
                    dset = epoch_parameters_grp.create_dataset(str(i), data=param)
                    dset.attrs['name'] = name
            
            # add all of the data
            data_grp = f.create_group("data")
            pred_labels_grp = f.create_group("pred_labels")
            pred_occs_grp = f.create_group("pred_occs")
            pred_deep_lifft_scores_grp = f.create_group("deep_lifft_scores")
            for data_subset_name in ['train', 'validation']:
                if data_subset_name == 'train' and not save_train_data: 
                    continue
                print "Saving %s data" % data_subset_name
                all_data = defaultdict(list)
                all_pred_labels = defaultdict(list)
                all_pred_occs = defaultdict(list)
                all_deep_lifft_scores = defaultdict(list)
                for data in self.iter_data(
                        100, data_subset_name, 
                        repeat_forever=False, 
                        include_chipseq_signal=True):
                    for key, values in data.iteritems():
                        all_data[key].append(values)
                    for key, values in self.predict_on_batch(data).iteritems():
                        all_pred_labels[key].append(values)
                    for key, values in self.predict_occupancies_on_batch(
                            data).iteritems():
                        all_pred_occs[key].append(values)
                    for key, values in self.predict_deeplifft_scores_on_batch(
                            data).iteritems():
                        all_deep_lifft_scores[key].append(values)

                data_subgrp = data_grp.create_group(data_subset_name)
                for key, values in all_data.iteritems():
                    values = np.concatenate(values, axis=0)
                    dset = data_subgrp.create_dataset(str(key), data=values)
                data_subgrp = pred_labels_grp.create_group(data_subset_name)
                for key, values in all_pred_labels.iteritems():
                    values = np.concatenate(values, axis=0)
                    dset = data_subgrp.create_dataset(str(key), data=values)
                data_subgrp = pred_occs_grp.create_group(data_subset_name)
                for key, values in all_pred_occs.iteritems():
                    values = np.concatenate(values, axis=0)
                    dset = data_subgrp.create_dataset(str(key), data=values)
                data_subgrp = pred_deep_lifft_scores_grp.create_group(data_subset_name)
                for key, values in all_deep_lifft_scores.iteritems():
                    values = np.concatenate(values, axis=0)
                    dset = data_subgrp.create_dataset(str(key), data=values)
        
        return 
        # code to pickle the actual model
        model._data_iterators = None
        import cPickle as pickle
        sys.setrecursionlimit(50000)
        rv = pickle.dumps(model)
        print rv[:10]
        assert False

    def evaluate(self, batch_size):
        # Print the validation results for this epoch:
        classification_results  = {}
        pred_prbs, labels, sample_ids = self.predict(batch_size)
        for key in pred_prbs.keys():
            for sample_index in xrange(sample_ids[key].shape[1]):
                sample_indices = np.array(
                    sample_ids[key][:,sample_index], dtype=bool)
                for index in xrange(labels[key].shape[1]):
                    if key in self._multitask_labels:
                        index_name = self._multitask_labels[key][index]
                    else:
                        index_name = str(index)
                    # skip purely ambiguous labels
                    if (labels[key][sample_indices,index] < -0.5).all():
                        continue
                    res = ClassificationResult(
                        labels[key][sample_indices,index], 
                        pred_prbs[key][sample_indices,index] > 0.5, 
                        pred_prbs[key][sample_indices,index]
                    )
                    try: sample_id = "sample%s-" % self.sample_ids[sample_index]
                    except IndexError: sample_id = ""
                    classification_results[
                        ("%s%s" % (
                            sample_id, key.split("_")[-1]), 
                         index_name)
                    ] = res
        return classification_results

    def train(self, samples_per_epoch, batch_size, nb_epoch, balanced=False):
        # Finally, launch the training loop.
        if VERBOSE: print("\n\n\n\n\nStarting training...")
        auPRCs = [0.0,]
        # We iterate over epochs:
        for epoch in xrange(nb_epoch):
            print 'auPRCs', epoch, auPRCs[-6:], max(auPRCs[-6:]), max(auPRCs)
            if max(auPRCs[-6:]) + 1e-6 < max(auPRCs):
                break
            
            #self._selex_penalty.set_value( 
            #    round(self._selex_penalty.get_value()/1.20, 6) )
            #self._chipseq_regularization_penalty.set_value( 
            #    round(self._chipseq_regularization_penalty.get_value()/1.20, 6))
            
            # In each epoch, we do a full pass over the training data:
            train_err = np.zeros(len(self._losses), dtype=float)
            progbar = Progbar(samples_per_epoch)
            for nb_train_batches_observed, data in enumerate(self.iter_data(
                    batch_size, 
                    'train', 
                    repeat_forever=True, 
                    shuffled=True,
                    balanced=balanced,
            )):
                if nb_train_batches_observed*batch_size > samples_per_epoch: 
                    break
                # we can use the values attriburte because iter_data  
                # returns an ordered dict
                filtered_data = [
                    data[key] for key in 
                    self._input_vars.keys() + self._target_vars.keys()
                ]
                err = self.train_fn(*filtered_data)
                if VERBOSE:
                    progbar.update(
                        nb_train_batches_observed*batch_size, 
                        [('train_err', err.sum()),]
                    )
                train_err += err
                        
            # calculate the test error
            validation_err = np.zeros(len(self._losses), dtype=float)
            validation_batches = 0
            for data in self.iter_data(
                    batch_size, 
                    'validation', # XXX 
                    repeat_forever=False, 
                    balanced=False, 
                    shuffled=False):
                # we can use the values attriburte because iter_data  
                # returns an ordered dict
                filtered_data = [
                    data[key] for key in 
                    self._input_vars.keys() + self._target_vars.keys()
                ]
                err = self.test_fn(*filtered_data)
                validation_err += err
                validation_batches += 1
            real_task_key = [
                key for key in self._losses.keys() 
                if key.startswith('invivo') and key.endswith('sequence')
            ]
            #assert len(real_task_key) == 1
            #self.validation_losses.append(
            #    dict(zip(
            #        self._losses.keys(), (validation_err/validation_batches))
            #     )[real_task_key[0]]
            #)

            # Print the validation results for this epoch:
            classification_results  = self.evaluate(batch_size)
            auPRC = 0.0
            for key, vals in sorted(classification_results.iteritems()):
                if not math.isnan(vals.auPRC):
                    auPRC += vals.auPRC
            print( 'val_err: %s' % zip(
                self._losses.keys(), (validation_err/validation_batches) ))
            print( 'mean auPRC: %.4f' % (auPRC/len(classification_results)))
            for key, vals in sorted(classification_results.iteritems()):
                print "-".join(key).ljust(40), vals
            self.validation_results.append(classification_results)
            auPRCs.append(auPRC/len(classification_results))
            
            params = OrderedDict()
            for i, param in enumerate(self.get_all_params()):
                params[(i, str(param))] = param.get_value().copy()
                #if str(param) == 'dnase_weights':
                #    print param, param.get_value()
                if str(param) == 'chem_affinity':
                    print param, param.get_value()
                if str(param) == 'dnase_weight':
                    print param, param.get_value()

            self.model_params.append(params)

        return

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Fit an rSeqDNN model')
    parser.add_argument(
        '--roadmap-sample-ids', 
        required=True,
        nargs='+',
        help='Roadmap sample ids to use.')
    parser.add_argument(
        '--validation-roadmap-sample-ids', 
        nargs='+',
        default=None,
        help='Roadmap sample id to validate on (ie dont train on this sample)')
    parser.add_argument(
        '--tf-names', 
        required=True,
        nargs='+',
        help='Transcription factors to use.')
    parser.add_argument(
        '--n-peaks',
        type=int,
        default=None,
        help='Maximum number of peaks to user per sample.')

    args = parser.parse_args()
    args.roadmap_sample_ids = sorted(
        set(args.roadmap_sample_ids + (
            args.validation_roadmap_sample_ids 
            if args.validation_roadmap_sample_ids is not None else []
        )))
    return args
    

def chipseq_main():        
    args = parse_args()

    # This is code that loads a small batch to catch errors 
    # early during debugging
    if False:
        pks = PartitionedSamplePeaksAndChIPSeqLabels(
            args.roadmap_sample_ids, 
            factor_names=args.tf_names, 
            max_n_samples=500, 
            validation_sample_ids=args.validation_roadmap_sample_ids
        ) 
        for batch in pks.iter_train_data(10):
            print batch.keys()
            for key in batch.keys():
                print key, batch[key].shape
            break
        tf_names = pks.factor_names
    model = JointBindingModel(
        args.n_peaks, 
        args.tf_names, 
        args.roadmap_sample_ids, 
        validation_sample_ids=args.validation_roadmap_sample_ids
    )

    model.train(
        args.n_peaks if args.n_peaks is not None else 300000, 
        500, 
        25, 
        balanced=False)
    model.save('Multitest.h5')
    return

def selex_main(n_samples, tf_name):
    model = JointBindingModel(
        n_samples, 
        invivo_factor_names=[], 
        sample_ids=[],
        invitro_factor_names=[tf_name,],
        use_three_base_encoding=False)
    model.train(
        n_samples if n_samples is not None else 100000, 500, 50, balanced=True)

if __name__ == '__main__':
    chipseq_main()
    #selex_main(5000, 'CTCF')
