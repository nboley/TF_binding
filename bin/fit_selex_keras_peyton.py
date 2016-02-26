import os, sys
import time

from collections import OrderedDict, defaultdict

import numpy as np

from pysam import FastaFile

from pyTFbindtools.cross_validation import ClassificationResult
 
from pyTFbindtools.DB import load_genome_metadata
from pyDNAbinding.sequence import one_hot_encode_sequence
from pyDNAbinding.misc import R, T

VERBOSE = True

import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates

import keras.backend as K

import theano
import theano.tensor as TT
 
from fit_selex_keras import SelexData
from lasagne.layers import Layer, InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, FlattenLayer, ExpressionLayer, GlobalPoolLayer
from lasagne import init
from lasagne.utils import create_param

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
                 use_three_base_encoding=True,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 **kwargs):
        super(ConvolutionDNASequenceBinding, self).__init__(input, **kwargs)
        self.use_three_base_encoding = use_three_base_encoding
        self.motif_len = motif_len
        self.nb_motifs = nb_motifs
        base_size = (3 if self.use_three_base_encoding else 4)
        filter_shape = (base_size, motif_len)
        self.W = self.add_param(
            W, (nb_motifs, 1, base_size, motif_len), name='W')
        self.b = self.add_param(
            b, (nb_motifs, ), name='b')

    def get_output_shape_for(self, input_shape):
        print "ConvolutionDNASequenceBinding", self.input_shape
        return (# number of obseravations
                self.input_shape[0],
                2,
                self.nb_motifs,
                # sequence length minus the motif length
                self.input_shape[3]-self.motif_len+1,
                )

    def get_output_for(self, input, **kwargs):
        print "ConvolutionDNASequenceBinding", self.input_shape
        if self.use_three_base_encoding:
            X_fwd = input[:,:,1:,:]
            X_rc = input[:,:,:3,:]
        else:
            X_fwd = input
            X_rc = input

        fwd_rv = K.conv2d(X_fwd, self.W, border_mode='valid') \
                 + TT.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rc_rv = K.conv2d(X_rc, self.W[:,::-1,:,::-1], border_mode='valid') \
                + TT.reshape(self.b, (1, self.nb_motifs, 1, 1))
        rv = TT.concatenate((fwd_rv, rc_rv), axis=2)
        return rv.dimshuffle((0,2,1,3))

def theano_calc_log_occs(affinities, chem_pot):
    inner = (-chem_pot+affinities)/(R*T)
    return TT.log(1.0 + TT.exp(inner))
    inner = (-chem_pot+affinities) #/(R*T)
    lower = TT.switch(inner<-10, TT.exp(inner), 0)
    mid = TT.switch((inner >= -10)&(inner <= 35), 
                    TT.log(1.0 + TT.exp(inner)),
                    0 )
    upper = TT.switch(inner>35, inner, 0)
    return -(lower + mid + upper)

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

class LogAnyBoundOcc(Layer):
    def get_output_shape_for(self, input_shape):
        #return self.input_shape
        return (input_shape[0], input_shape[1], input_shape[2], 1)

    def get_output_for(self, input, **kwargs):
        log_none_bnd = TT.sum(
            TT.log(1-TT.clip(TT.exp(input), 1e-6, 1-1e-6)), axis=3, keepdims=True)
        at_least_1_bnd = 1-TT.exp(log_none_bnd)
        max_occ = TT.max(TT.exp(input), axis=3, keepdims=True)
        # we take the weighted sum because the max is easier to fit, and 
        # thus this helps to regularize the optimization procedure
        rv = TT.log(0.05*max_occ + 0.95*at_least_1_bnd)
        return rv

class OccMaxPool(Layer):
    def __init__(self, input, num_tracks, num_bases, **kwargs):
        self.num_tracks = num_tracks
        self.num_bases = num_bases
        super(OccMaxPool, self).__init__(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        assert input_shape[1] == 1
        num_output_tracks = (
            1 if self.num_tracks == 'full' 
            else input_shape[2]//self.num_tracks )
        num_output_bases = (
            1 if self.num_bases == 'full' 
            else input_shape[3]//self.num_bases )
        return (
            input_shape[0],
            1,
            num_output_tracks, # + (
            #    1 if input_shape[1]%self.num_tracks > 0 else 0),
            num_output_bases# + (
            #    1 if input_shape[3]%self.num_bases > 0 else 0)
        )

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
        X = input
        #X = X.dimshuffle((0,2,1,3))
        rv = K.pool2d(
            X, 
            pool_size=(num_tracks, num_bases), 
            strides=(num_tracks, num_bases),
            pool_mode='max'
        )
        #rv =rv.dimshuffle((0,2,1,3))

        """
        X = K.permute_dimensions(input, (0,2,1,3))
        rv = TT.signal.downsample.max_pool_2d(
            X, 
            ds=(num_tracks, num_bases), 
            st=(num_tracks, num_bases),
            mode='max',
            ignore_border=True
        )
        rv = K.permute_dimensions(rv, (0,2,1,3))
        """
        return rv


class JointBindingModel():
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
        
        network = ConvolutionDNASequenceBinding(
            network, nb_motifs=4, motif_len=16)
        network = LogNormalizedOccupancy(network, 0.0)
        network = LogAnyBoundOcc(network)
        network = OccMaxPool(network, 'full', 'full' )
        #network = GlobalPoolLayer(network, TT.max) 
        #network = ExpressionLayer(network, TT.exp)
        #network = FlattenLayer(network)
        #network = DenseLayer(network, 1)
        
        #network = Conv2DLayer(network,2, (1,16))
        #network = DenseLayer(network, 1)

        self._networks[name + ".output"] = network
        self._data_iterators[name] = selex_experiment.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(lasagne.objectives.squared_error(prediction, target_var))
        self._losses[name] = loss
    
    def add_selex_experiments(self):
        for exp in self.selex_experiments:
            self.add_selex_experiment(exp)

    def add_chipseq_regularization(
            self, occupancies_input_name, name_prefix, output_dim):
        seq_length = 1000
        
        input_var = TT.tensor4(name + '.fwd_seqs')
        self._input_vars[name + '.fwd_seqs'] = input_var
        target_var = TT.matrix(name + '.output')
        self._target_vars[name + '.output'] = target_var
        
        network = InputLayer(
            shape=(None, 1, 4, seq_length), input_var=input_var)

        network = LogAnyBoundOcc(network)
        network = OccMaxPool(network, 4)
        network = FlattenLayer(network)

        self._networks[name + ".output"] = network
        self._data_iterators[name] = selex_experiment.iter_batches

        prediction = lasagne.layers.get_output(network)
        loss = TT.mean(lasagne.objectives.squared_error(prediction, target_var))
        self._losses[name] = loss

    def _build(self):
        # build the predictions dictionary
        total_loss = TT.sum(self._losses.values())
        params = lasagne.layers.get_all_params(
            self._networks.values(), trainable=True)
        updates = lasagne.updates.adam(total_loss, params)

        # build the training function
        all_vars = self._input_vars.values() + self._target_vars.values()
        self.train_fn = theano.function(
            all_vars, total_loss, updates=updates)
        
        # build the prediction functions
        self.predict_fns = {}
        for key in self._target_vars:
            predict_fn = theano.function(
                self._input_vars.values(), 
                lasagne.layers.get_output(
                    self._networks[key], deterministic=True),
                on_unused_input='ignore'
            )
            self.predict_fns[key] = predict_fn
        

    def __init__(self, n_samples, factor_names):
        self._losses = {}
        
        self._input_vars = OrderedDict()
        self._target_vars = OrderedDict()
        self._networks = OrderedDict()
        self._data_iterators = OrderedDict()
        
        self.selex_experiments = SelexData(n_samples)
        for factor_name in factor_names:
            self.selex_experiments.add_all_selex_experiments_for_factor(
                factor_name)
        self.add_selex_experiments()
        
        self._build()

    def iter_data(self, batch_size, data_subset, repeat_forever):
        iterators = OrderedDict()
        for key, iter_inst in self._data_iterators.iteritems():
            iterators[key] = iter_inst(batch_size, data_subset, repeat_forever)

        while True:
            merged_output = {}
            for name_prefix, iterator in iterators.iteritems():
                data_dict = next(iterator)
                for key, data in data_dict.iteritems():
                    assert name_prefix + "." + key not in merged_output
                    merged_output[name_prefix + "." + key] = data
            yield merged_output
            
        return

    def predict_on_batch(self, data):
        predictions = {}
        for output_key, fn in self.predict_fns.iteritems():
            predictions[output_key] = fn(
                *[data[input_key] for input_key in self._input_vars.keys()])
        return predictions
            
    def predict(self, batch_size):
        data_iterator = self.iter_data(batch_size, 'validation', False)
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
        
    def train(self, batch_size, num_epochs):
        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for data in self.iter_data(batch_size, 'train', False):
                train_data = [
                    data[input_key] for input_key in self._input_vars.keys()
                ] + [ 
                    data[output_key] for output_key in self._target_vars.keys() 
                ]
                err = self.train_fn(*train_data)
                train_err += err
                train_batches += 1
            
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            pred_prbs, labels = self.predict(batch_size)
            for key in pred_prbs.keys():
                print key.ljust(40), ClassificationResult(
                    labels[key], pred_prbs[key] > 0.5, pred_prbs[key])
        
        pass

def main():    
    model = JointBindingModel(100000, ['CTCF',]) # 'MAX'])
    model.train(1000, 500)
    return

main()
