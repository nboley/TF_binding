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

import keras.backend as K
from keras.utils.generic_utils import Progbar

import theano
import theano.tensor as TT
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.pool import Pool
from theano.tensor.nnet import sigmoid, ultra_fast_sigmoid, binary_crossentropy, softmax, softplus
 
from pyTFbindtools.peaks import SelexData, PartitionedSamplePeaksAndLabels

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

def subsample_batch(model, batch, subsample_sizes):
    output = OrderedDict()
    pred_vals = model.predict_on_batch(batch)
    for key in pred_vals:
        batch_size = subsample_sizes[key]

        losses = (pred_vals[key] - batch[key])**2
        weights = np.zeros_like(losses)
        
        # set ambiguous labels to 0
        losses[batch[key] < -0.5] = 0
        # add to the losses,
        weights = losses.sum(1)
        # re-weight the rows with ambiguous labels
        weights *= losses.shape[1]/((
            batch[key] > -0.5).sum(1) + 1e-6)
        # make sure that every row has some weight
        weights += 1e-6
        # normalize the weights to 1
        weights /= weights.sum()

        # choose the max batch sizes random indices
        loss_indices = np.random.choice(
            len(weights), size=subsample_sizes[key], replace=False, p=weights)
        output[key] = batch[key][loss_indices,:]
    return output

def iter_balanced_batches(batch_iterator):
    labels = None
    batch_size = subsample_sizes[key]
    weights = np.ones(batch_size)

    losses = (pred_vals[key] - batch[key])**2
    # set ambiguous labels to 0
    losses[batch[key] < -0.5] = 0
    # add to the losses,
    inner_weights = losses.sum(1)
    # re-weight the rows with ambiguous labels
    inner_weights *= losses.shape[1]/((
        batch[key] > -0.5).sum(1) + 1e-6)
    # make sure that every row has some weight
    inner_weights += 1e-6
    weights += inner_weights
    # normalize the weights to 1
    weights /= weights.sum()

    # choose the max batch sizes random indices
    loss_indices = np.random.choice(
        len(weights), size=subsample_sizes[key], replace=False, p=weights)
    output[key] = batch[key][loss_indices,:]

    return output

def iter_weighted_batch_samples(
        model, batch_iterator, oversampling_ratio=1, callback=subsample_batch):
    """Iter batches where poorly predicted samples are more frequently selected.

    model: the model to predict from
    batch_iterator: batch iterator to draw samples from
    oversampling_ratio: how many batch_iterator batches to sample from
    """
    assert oversampling_ratio > 0

    while True:
        # group the output of oversampling_ratio batches together
        all_batches = OrderedDict()
        batch_sizes = OrderedDict()
        for key, val in next(batch_iterator).iteritems():
            all_batches[key] = [val,]
            batch_sizes[key] = val.shape[0]
        for i in xrange(oversampling_ratio-1):
            for key, val in next(batch_iterator).iteritems():
                all_batches[key].append(val)
        # stack the output
        for key, batches in all_batches.iteritems():
            all_batches[key] = np.vstack(batches)

        yield callback(model, all_batches, batch_sizes)
    
    return

def mse_skip_ambig(y_true, y_pred):
    #return (y_true - y_pred)**2
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    return K.square(y_pred*non_ambig - y_true*non_ambig)*y_true.shape[0]/cnts


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

    return (-1e-6 -(1+beta*beta)*precision*recall)/(beta*beta*precision+recall+2e-6)

def expected_F1_skip_ambig(y_true, y_pred, beta=0.5):
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    rv = expected_F1_loss(
        TT.clip(y_pred*non_ambig, 1e-6, 1-1e-6), 
        TT.clip(y_true*non_ambig, 1e-6, 1-1e-6),
        beta
    )
    return rv*y_true.shape[0]/cnts

global_loss_fn = cross_entropy_skip_ambig #mse_skip_ambig #expected_F1_skip_ambig

def load_data(fname):
    cached_fname = "peytons.cachedseqs.obj"
    try:
        with open(cached_fname) as fp:
            print "Loading cached seqs"
            obj = np.load(fp)
            return (obj['ts'], obj['tl']), (obj['vs'], obj['vl'])
    except IOError:
        pass

    genome_fasta = FastaFile('hg19.genome.fa')
    with open(fname) as fp:
        num_lines = sum(1 for line in fp)
    with open(fname) as fp:
        n_cols = len(next(iter(fp)).split())
    pk_width = 1000

    n_train = 0
    train_seqs = 0.25 * np.ones((num_lines, pk_width, 4), dtype='float32')
    train_labels = np.zeros((num_lines, n_cols-4), dtype='float32')
    
    n_validation = 0
    validation_seqs = 0.25 * np.ones((num_lines, pk_width, 4), dtype='float32')
    validation_labels = np.zeros((num_lines, n_cols-4), dtype='float32')
    
    with open(fname) as fp:
        for i, line in enumerate(fp):
            if i%1000 == 0: print i
            if i == 0: continue
            data = line.split()
            seq = genome_fasta.fetch(data[0], int(data[1]), int(data[2]))
            if len(seq) != pk_width: continue
            if data[0] in ('chr3', 'chr4'): 
                # dont include jittered peaks in the test set
                if i > 100000 and i < 300000:
                    continue
                # skip testing for now
                continue
            elif data[0] in ('chr1', 'chr2'): 
                # dont include jittered peaks in the validation set
                if i > 100000 and i < 300000:
                    continue
                pk_index = n_validation
                n_validation += 1
                rv = validation_seqs
                labels = validation_labels
            else:
                pk_index = n_train
                n_train += 1
                rv = train_seqs
                labels = train_labels
            coded_seq = one_hot_encode_sequence(seq)
            rv[pk_index,:,:] = coded_seq
            labels[pk_index,:] = np.array([float(x) for x in data[4:]])
    
    train_seqs = train_seqs[:n_train,:,:]
    train_labels = train_labels[:n_train]
    
    validation_seqs = validation_seqs[:n_validation,:,:]
    validation_labels = validation_labels[:n_validation]

    with open(cached_fname, "w") as ofp:
        print "Saving seqs"
        np.savez(
            ofp, 
            ts=train_seqs, tl=train_labels, 
            vs=validation_seqs, vl=validation_labels)

    return (train_seqs, train_labels), (validation_seqs, validation_labels)

class ConvolutionDNASequenceBinding(Layer):
    def __init__(self, 
                 input,
                 nb_motifs, motif_len, 
                 use_three_base_encoding=False,
                 W=init.HeNormal(), b=init.Constant(-3.0),
                 **kwargs):
        super(ConvolutionDNASequenceBinding, self).__init__(input, **kwargs)
        self.use_three_base_encoding = use_three_base_encoding
        self.motif_len = motif_len
        self.nb_motifs = nb_motifs
        base_size = (3 if self.use_three_base_encoding else 4)
        filter_shape = (base_size, motif_len)
        self.W = self.add_param(
             W, (nb_motifs, 1, base_size, motif_len), name='W')
        if use_three_base_encoding:
            self.b = self.add_param(
                b, (nb_motifs, ), name='b')
        else:
            self.b = None
    
    def get_output_shape_for(self, input_shape):
        return (# number of obseravations
                self.input_shape[0],
                2,
                self.nb_motifs,
                # sequence length minus the motif length
                self.input_shape[3],
                )

    def get_output_for(self, input, **kwargs):
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

def theano_calc_log_occs(affinities, chem_pot):
    inner = (-chem_pot+affinities)/(R*T)
    return -TT.log(1.0 + TT.exp(inner))
    lower = TT.switch(inner<-10, TT.exp(inner), 0)
    mid = TT.switch((inner >= -10)&(inner <= 35), 
                    TT.log(1.0 + TT.exp(inner)),
                    0 )
    upper = TT.switch(inner>35, inner, 0)
    return -(lower + mid + upper)


class StackStrands(Layer):
    """Stack strand independent convolutions.

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
    
class LogNormalizedOccupancy(Layer):
    def __init__(
            self, 
            input,
            init_chem_affinity=0.0, 
            steric_hindrance_win_len=None, 
            **kwargs):
        super(LogNormalizedOccupancy, self).__init__(input, **kwargs)
        self.init_chem_affinity = init_chem_affinity
        self.chem_affinity = self.add_param(
            init.Constant(self.init_chem_affinity),
            (), 
            name='chem_affinity')
        self.steric_hindrance_win_len = (
            0 if steric_hindrance_win_len is None 
            else steric_hindrance_win_len
        )

    def get_output_shape_for(self, input_shape):
        #return self.input_shape
        assert self.input_shape[1] == 2
        return (# number of obseravations
                self.input_shape[0], 
                1,
                2*self.input_shape[2],
                # sequence length minus the motif length
                self.input_shape[3] #-2*(self.steric_hindrance_win_len-1)
        )
    
    def get_output_for(self, input, **kwargs):
        X = input #self.get_input(train)
        X = X.reshape(
            (X.shape[0], 1, 2*X.shape[2], X.shape[3]))
        # calculate the log occupancies
        log_occs = theano_calc_log_occs(-X, self.chem_affinity)
        # reshape the output so that the forward and reverse complement 
        # occupancies are viewed as different tracks 
        if self.steric_hindrance_win_len == 0:
            log_norm_factor = 0
        else:
            # correct occupancies for overlapping binding sites
            occs = K.exp(log_occs)
            kernel = K.ones((1, 1, 1, 2*self.steric_hindrance_win_len-1), dtype='float32')
            win_occ_sum = K.conv2d(occs, kernel, border_mode='same').sum(axis=2, keepdims=True)
            win_prb_all_unbnd = TT.exp(
                K.conv2d(K.log(1-occs), kernel, border_mode='same')).sum(axis=2, keepdims=True)
            log_norm_factor = TT.log(win_occ_sum + win_prb_all_unbnd)
        #start = max(0, self.steric_hindrance_win_len-1)
        #stop = min(self.output_shape[3], 
        #           self.output_shape[3]-(self.steric_hindrance_win_len-1))
        #rv = log_occs[:,:,:,start:stop] - log_norm_factor
        rv = (log_occs - log_norm_factor)
        return rv

class AnyBoundOcc(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

    def get_output_for(self, input, **kwargs):
        input = TT.clip(input, 1e-6, 1-1e-6)
        #return softmax(input)
        #return TT.clip(input.sum(axis=3, keepdims=True), 1e-6, 1-1e-6)
        log_none_bnd = TT.sum(
            TT.log(1-input), axis=3, keepdims=True)
        at_least_1_bnd = 1-TT.exp(log_none_bnd)
        return at_least_1_bnd
        ## we take the weighted sum because the max is easier to fit, and 
        ## thus this helps to regularize the optimization procedure
        max_occ = TT.max(input, axis=3, keepdims=True)
        return max_occ # + 0.5*at_least_1_bnd

class OccupancyLayer(Layer):
    def __init__(
            self, 
            input,
            init_chem_affinity=0.0,
            dnase_signal=None,
            **kwargs):
        super(OccupancyLayer, self).__init__(input, **kwargs)
        self.dnase_signal = dnase_signal
        if dnase_signal is not None:
            self.dnase_weight = self.add_param(
                init.Constant(1e-1),
                (self.input_shape[2],), 
                name='dnase_weight'
            )
        
        self.chem_affinity = self.add_param(
            init.Constant(init_chem_affinity),
            (self.input_shape[2],), 
            name='chem_affinity'
        )

    def get_output_shape_for(self, input_shape):
        return self.input_shape

    def get_output_for(self, input, **kwargs):
        """
        inner = (-chem_pot+affinities)/(R*T)
        return -TT.log(1.0 + TT.exp(inner))
        lower = TT.switch(inner<-10, TT.exp(inner), 0)
        mid = TT.switch((inner >= -10)&(inner <= 35), 
                         TT.log(1.0 + TT.exp(inner)),
                        0 )
        upper = TT.switch(inner>35, inner, 0)
        return -(lower + mid + upper)
        """
        rv = self.chem_affinity[None, None, :, None] + input
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
        #self.affinity_weights = self.add_param(
        #    init.Constant(0.5),
        #    (self.n_tracks,), 
        #    name='affinity_weights'
        #)
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
        return (
            affinities #*self.affinity_weights[None,:,None]  
            + track_access*self.dnase_weights[None,None,:,None]
        )

class JointBindingModel():    
    def _init_shared_affinity_params(self, use_three_base_encoding):
        # initialize the full subdomain convolutional filter
        self.num_invivo_convs = 0
        self.num_tf_specific_invitro_affinity_convs = 8
        self.num_tf_specific_invivo_affinity_convs = 0 # HERE 
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
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        
        network = InputLayer(
            shape=(None, 1, 4, seq_length), input_var=input_var)
        
        W, b = self.get_tf_specific_affinity_params(
            selex_experiment.factor_name, only_invitro=True)
        network = ConvolutionDNASequenceBinding(
            network, 
            nb_motifs=self.num_tf_specific_invitro_affinity_convs, 
            motif_len=self.affinity_conv_size,
            use_three_base_encoding=self._use_three_base_encoding,
            W=W, 
            b=b)
        network = StackStrands(network)
        network = OccupancyLayer(network, -6.0)
        network = OccMaxPool(network, 'full', 'full' )
        network = FlattenLayer(network)
        
        self._networks[name + ".output"] = network
        self._data_iterators[name] = selex_experiment.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(lasagne.objectives.squared_error(prediction, target_var))
        self._losses[name] = self._selex_penalty*loss

    def add_selex_experiments(self, selex_experiments):
        for exp in selex_experiments:
            self.add_selex_experiment(exp)
    
    def add_chipseq_trace_model(self, pks_and_labels):
        name = 'invivo_%s_sequence' % pks_and_labels.sample_id 

        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        self._multitask_labels[name + '.output'] = pks_and_labels.factor_names
        #cov_target_var = TT.matrix(name + '.chipseq_cov')
        #self._target_vars[name + '.chipseq_cov'] = cov_target_var
        #self._multitask_labels[name + '.chipseq_cov'] = pks_and_labels.factor_names
        
        network = InputLayer(
            shape=(None, 1, 4, pks_and_labels.seq_length), input_var=input_var)

        network = ConvolutionDNASequenceBinding(
            network, 
            nb_motifs=self.num_affinity_convs, 
            motif_len=self.affinity_conv_size, 
            use_three_base_encoding=self._use_three_base_encoding,
            W=self.affinity_conv_filter, 
            b=self.affinity_conv_bias)
        network = LogNormalizedOccupancy(network, -6.0)
        #network = LogAnyBoundOcc(network)
        network = OccMaxPool(network, 2*self.num_tf_specific_convs, 1)
        network = ExpressionLayer(network, TT.exp)
        #occs_prediction = lasagne.layers.get_output(network)

        network = OccMaxPool(network, 'full', 'full')        
        network = FlattenLayer(network)
        #network = DenseLayer(network, pks_and_labels.labels.shape[1])

        prediction = lasagne.layers.get_output(network) 
        label_loss = TT.mean(TT.sum(global_loss_fn(target_var, prediction), axis=-1))
        cov_loss = 0 #TT.sum((TT.flatten(cov_target_var)-TT.flatten(occs_prediction))**2)
        self._networks[name + ".output"] = network
        self._data_iterators[name] = pks_and_labels.iter_batches
        self._losses[name] = label_loss + cov_loss

        return

    def add_simple_chipseq_model(self, pks_and_labels):
        name = 'invivo_%s_sequence.simple' % pks_and_labels.sample_id 

        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        self._multitask_labels[name + '.output'] = pks_and_labels.factor_names
        
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

        self._networks[name + ".output"] = network

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
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        self._multitask_labels[name + '.output'] = pks_and_labels.factor_names

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
        self._add_chipseq_regularization(single_tf_affinities, target_var)

        if include_dnase is True:
            access_input_var = TT.tensor4(name + '.dnase_cov')
            self._input_vars[name + '.dnase_cov'] = access_input_var
            dnase = InputLayer(
                shape=(None, 1, 1, pks_and_labels.seq_length), 
                input_var=access_input_var
            )
            network = ConvolveDNASELayer(single_tf_affinities, dnase)
            #network = ConcatLayer([access, network], axis=2)
            #network = Conv2DLayer(
            #    network, 
            #    2*self.num_affinity_convs,
            #    (2*self.num_affinity_convs+1,1),
            #    nonlinearity=lasagne.nonlinearities.identity
            #)
            #network = DimshuffleLayer(network, (0,2,1,3))

        #network = Conv2DLayer(
        #    network, 
        #    2*self.num_affinity_convs,
        #    (2*self.num_affinity_convs,1),
        #    nonlinearity=(softplus if USE_SOFTPLUS_ACTIVATION 
        #                  else lasagne.nonlinearities.identity)
        #)
        #network = DimshuffleLayer(network, (0,2,1,3))
        #if include_dnase is True:
        #    network = ConvolveDNASELayer(network, access)
        
        network = OccMaxPool(network, 1, 32, 4)
        network = Conv2DLayer(
            network, 
            2*self.num_affinity_convs,
            (2*self.num_affinity_convs,16),
            nonlinearity=(softplus if USE_SOFTPLUS_ACTIVATION 
                          else lasagne.nonlinearities.identity),
        )
        cobinding_penalty = 0 #regularize_layer_params(network,l1)
        network = DimshuffleLayer(network, (0,2,1,3))
        if include_dnase is True:
            grpd_dnase = OccMaxPool(dnase, 1, 92, 4)
            network = ConvolveDNASELayer(network, grpd_dnase)

        network = OccupancyLayer(
            network, 
            init_chem_affinity=-8.0, 
            dnase_signal=None #TT.log(1+TT.max(access_input_var, axis=-1)).flatten()
        )
        self._occupancy_layers[name + ".occupancy"] = network
        network = OccMaxPool(network, 2*self.num_tf_specific_convs, 8)
        
        network = AnyBoundOcc(network)
        network = OccMaxPool(network, 1, 'full')
        
        #network = DenseLayer(
        #    network, len(self.invivo_factor_names), nonlinearity=sigmoid)
        
        network = FlattenLayer(network)
        self._networks[name + ".output"] = network
        self._data_iterators[name] = pks_and_labels.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction))
        self._losses[name] = loss
        return


    def add_DIGN_chipseq_samples(self, pks_and_labels, include_DNASE=True): 
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_id
        name = 'invivo_%s_DIGN_sequence' % pks_and_labels.sample_id 
        
        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        self._multitask_labels[name + '.output'] = pks_and_labels.factor_names

        n_convs = 25*len(self.factor_names)
        
        network = InputLayer(
            shape=(None, 1, 4, pks_and_labels.seq_length), input_var=input_var)

        if include_DNASE:
            access_input_var = TT.tensor4(name + '.dnase_cov')
            self._input_vars[name + '.dnase_cov'] = access_input_var
            access = InputLayer(
                shape=(None, 1, 1, pks_and_labels.seq_length),
                input_var=access_input_var
            )
            access = ExpressionLayer(access, lambda x: TT.log(
                1e-12+x/TT.max(x, keepdims=True)))
            network = ConcatLayer([access, network], axis=2)
            network = Conv2DLayer(
                network, n_convs, (5,15),
                nonlinearity=lasagne.nonlinearities.rectify)
        else:
            network = Conv2DLayer(
                network, n_convs, (4,15),
                nonlinearity=lasagne.nonlinearities.rectify)

        network = DropoutLayer(network, 0.2)
        network = Conv2DLayer(
            network, n_convs, (1,15),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = DropoutLayer(network, 0.2)
        network = Conv2DLayer(
            network, n_convs, (1,15),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = DropoutLayer(network, 0.2)
        network = DimshuffleLayer(network, (0,2,1,3))
        network = MaxPool2DLayer(network, (1, 35))
        network = DenseLayer(
            network, 
            len(pks_and_labels.factor_names), 
            nonlinearity=lasagne.nonlinearities.sigmoid)

        self._networks[name + ".output"] = network
        self._data_iterators[name] = pks_and_labels.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(global_loss_fn(target_var, prediction)) + cobinding_penalty
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
        #updates = lasagne.updates.apply_momentum(updates, params, 0.9)
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
                 invitro_factor_names=[],
                 use_three_base_encoding=True):
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
        #self.add_selex_experiments(selex_experiments)

        self._chipseq_regularization_penalty = create_param(
            lambda x: 2.0, (), 'chipseq_penalty')
        pks = PartitionedSamplePeaksAndLabels(
            sample_ids, factor_names=invivo_factor_names, n_samples=n_samples)
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
            include_dnase_signal=True
        )
        pred_prbs = defaultdict(list)
        labels = defaultdict(list)
        for i, data in enumerate(data_iterator):
            for key, prbs in self.predict_on_batch(data).iteritems():
                pred_prbs[key].append(prbs)
                labels[key].append(data[key])
                
        # stack everything back together
        for key in pred_prbs.iterkeys():
            pred_prbs[key] = np.vstack(pred_prbs[key])
            labels[key] = np.vstack(labels[key])

        return pred_prbs, labels

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
            include_chipseq_signal=True, 
            include_dnase_signal=True))
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
                        include_dnase_signal=True, 
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
        pred_prbs, labels = self.predict(batch_size)
        for key in pred_prbs.keys():
            for index in xrange(labels[key].shape[1]):
                if key in self._multitask_labels:
                    index_name = self._multitask_labels[key][index]
                else:
                    index_name = str(index)
                res = ClassificationResult(
                    labels[key][:,index], 
                    pred_prbs[key][:,index] > 0.5, 
                    pred_prbs[key][:,index]
                )
                classification_results[(key, index_name)] = res
        return classification_results

    def train(self, samples_per_epoch, batch_size, nb_epoch, balanced=False):
        # Finally, launch the training loop.
        if VERBOSE: print("\n\n\n\n\nStarting training...")
        # We iterate over epochs:
        for epoch in xrange(nb_epoch):
            self._selex_penalty.set_value( 
                round(self._selex_penalty.get_value()/1.20, 6) )
            self._chipseq_regularization_penalty.set_value( 
                round(self._chipseq_regularization_penalty.get_value()/1.20, 6))
            
            # In each epoch, we do a full pass over the training data:
            train_err = np.zeros(len(self._losses), dtype=float)
            progbar = Progbar(samples_per_epoch)
            for nb_train_batches_observed, data in enumerate(self.iter_data(
                    batch_size, 
                    'train', 
                    repeat_forever=True, 
                    shuffled=True,
                    balanced=balanced,
                    include_dnase_signal=True
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
                    shuffled=False,
                    include_dnase_signal=True):
                # we can use the values attriburte because iter_data  
                # returns an ordered dict
                filtered_data = [
                    data[key] for key in 
                    self._input_vars.keys() + self._target_vars.keys()
                ]
                err = self.test_fn(*filtered_data)
                validation_err += err
                validation_batches += 1
            print( 'val_err: %s' % zip(
                self._losses.keys(), (validation_err/validation_batches) ))
            real_task_key = [
                key for key in self._losses.keys() 
                if key.startswith('invivo') and key.endswith('sequence')
            ]
            assert len(real_task_key) == 1
            self.validation_losses.append(
                dict(zip(
                    self._losses.keys(), (validation_err/validation_batches))
                 )[real_task_key[0]]
            )

            # Print the validation results for this epoch:
            classification_results  = self.evaluate(batch_size)
            for key, vals in sorted(classification_results.iteritems()):
                print "-".join(key).ljust(40), vals
            self.validation_results.append(classification_results)
            
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
            #print "Affinity Conv Bias: %.2f-%.2f" % (
            #    self.affinity_conv_bias.get_value() 
            #    + self.affinity_conv_filter.get_value().min(axis=2).sum(axis=-1),
            #    self.affinity_conv_bias.get_value() 
            #    + self.affinity_conv_filter.get_value().max(axis=2).sum(axis=-1)
            #)
            
        pass

def load_chipseq_reads(bam_fps):
    factor_grpd_reads = defaultdict(list)
    for fp in bam_fps:
        reads = ChIPSeqReads(fp.name).init()
        factor_grpd_reads[reads.factor].append(reads)
    
    for factor in factor_grpd_reads.iterkeys():
        factor_grpd_reads[factor] = MergedReads(factor_grpd_reads[factor])
    
    return dict(factor_grpd_reads)

def single_sample_main():
    tf_name = sys.argv[1]
    sample_id = sys.argv[2]
    try: 
        n_samples = int(sys.argv[3])
    except IndexError: 
        n_samples = None

    pks = PartitionedSamplePeaksAndLabels(
        sample_id, factor_names=[tf_name,], n_samples=n_samples)
    rv = next(pks.iter_batches(
        100, 'train', False, include_chipseq_signal=False, include_dnase_signal=True))
    #return
    model = JointBindingModel(
        n_samples, 
        [tf_name,], 
        [sample_id,],
    #    ['YY1', 'CTCF', 'MAX', 'RFX5', 'USF1', 'PU1', 'NFE2', 'ATF4', 'ATF7'])
        [tf_name,],
        use_three_base_encoding=False)
    #model.predict_deeplifft_scores_on_batch(
    #    next(model.iter_data(100, 'train', False, include_dnase_signal=True)))
    #assert False

    model.train(
        n_samples if n_samples is not None else 100000, 100, 5, balanced=True)
    model.train(
        n_samples if n_samples is not None else 100000, 100, 10, balanced=False)
    model.save('model.%s.%s.%i.h5' % (tf_name, sample_id, n_samples))
    return

def all_tfs_main():
    sample_id = sys.argv[1]
    try: 
        n_samples = int(sys.argv[2])
    except IndexError: 
        n_samples = None
    sample_ids = [sample_id,]
    pks = PartitionedSamplePeaksAndLabels(
        sample_id, factor_names=None, n_samples=5000)
    print len(pks.factor_names)
    print pks.factor_names
    print pks.train.labels.shape
    #return
    model = JointBindingModel(300000, pks.factor_names, sample_ids)
    model.train(300000, 100, 30)
    model.save('Multitask.%s.h5' % sample_id)
    return

def many_tfs_main():
    tf_names = sys.argv[1].split(",")
    sample_ids = sys.argv[2].split(",")
    try: 
        n_samples = int(sys.argv[3])
    except IndexError: 
        n_samples = None
    print tf_names, sample_ids, n_samples
    pks = PartitionedSamplePeaksAndLabels(
        sample_ids, factor_names=tf_names, n_samples=5000)
    for batch in pks.iter_train_data(100):
        break
    model = JointBindingModel(n_samples, pks.factor_names, sample_ids)
    model.train(n_samples, 100, 60)
    model.save('Multitask.%s.%s.h5' % (sample_id, "_".join(tf_names)))
    

def main():        
    #tf_names = [x[1] for x in SelexData.find_all_selex_experiments()]
    tf_names = ['CTCF', 'MAX', 'MYC', 'YY1'] # 'MAX', 'MYC', 'YY1'] #'MYC', 'YY1'] # 'MAX', 'BHLHE40']
    sample_ids = ['E123',]
    model = JointBindingModel(50000, tf_names, sample_ids)
    model.train(10000, 100, 10)
    model.save('Multitest.h5')
    return

def test_chipseq():
    from pyTFbindtools.peaks import load_chipseq_coverage, build_peaks_label_mat
    tf_name = sys.argv[1]
    tf_id = 'T044268_1.02'
    sample_id = sys.argv[2]
    #pks, tf_ids, labels = build_peaks_label_mat(1, sample_id, 500)
    #load_chipseq_coverage(sample_id, x, pks)
    #from grit.lib.multiprocessing_utils import run_in_parallel
    #run_in_parallel(
    #    8, #len(tf_ids), 
    #    lambda x: load_chipseq_coverage(sample_id, x, pks),
    #    [[x,] for x in tf_ids]
    #)
    n_samples = 1000
    pks = PartitionedSamplePeaksAndLabels(
        sample_id, factor_names=[tf_name,], n_samples=n_samples)
    print pks.data.chipseq_coverage.shape

#test_chipseq()

if __name__ == '__main__':
    #main()
    #single_sample_main()
    #all_tfs_main()
    many_tfs_main()
