import sys
import hashlib
import re

import functools
from collections import defaultdict
from random import shuffle, sample

sys.path.insert(0, '/users/nasa/FeatureExtractionTools/bigWigFeaturize/')
import bigWigFeaturize
from collections import namedtuple
Region = namedtuple('Region', ['chrom', 'start', 'stop'])

import cPickle as pickle

import numpy as np
from scipy.stats import itemfreq

from pysam import FastaFile

from pyTFbindtools.peaks import (
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB, 
    encode_peaks_sequence_into_binary_array, 
    PeaksAndLabels,
    build_peaks_label_mat, iter_summit_centered_peaks)
from pyTFbindtools.DB import load_genome_metadata
from pyTFbindtools.cross_validation import ClassificationResult

from fit_selex import SelexDBConn, load_fastq, optional_gzip_open
from pyDNAbinding.binding_model import (
    FixedLengthDNASequences, 
    ConvolutionalDNABindingModel, 
    EnergeticDNABindingModel
)
from pyDNAbinding.plot import plot_bases, pyplot
from pyDNAbinding.sequence import one_hot_encode_sequence
from pyDNAbinding.keraslib import *

VERBOSE = False

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
    non_ambig = (y_true > -0.5)
    cnts = non_ambig.sum(axis=0, keepdims=True)
    return K.mean(K.square(y_pred*non_ambig - y_true*non_ambig)*y_true.shape[0]/cnts, axis=-1) 

def weighted_loss_generator(weight, loss_fn):
    def f(y_true, y_pred):
        return weight*loss_fn(y_true, y_pred)
    return f

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

def calc_val_results(model, batch_size):
    data_iterator = model.iter_validation_data(
        batch_size, repeat_forever=False)
    pred_prbs = defaultdict(list)
    labels = defaultdict(list)
    for i, data in enumerate(data_iterator):
        for key, prbs in model.predict_on_batch(data).iteritems():
            # make sure every set of predicted prbs have an associated key
            assert key in data, "Predicted prbs don't have associated labels"
            pred_prbs[key].append(prbs)
            labels[key].append(cast_if_1D(data[key]))


    val_results = {}
    assert set(pred_prbs.keys()) == set(labels.keys())
    for key in pred_prbs.keys():
        inner_prbs = np.concatenate(pred_prbs[key], axis=0)
        inner_labels = np.concatenate(labels[key], axis=0)

        assert len(inner_prbs.shape) == 2
        for i in xrange(inner_prbs.shape[1]):
            prbs_subset = inner_prbs[:,i].ravel()
            classes = np.array(prbs_subset > 0.5, dtype=int)
            truth = inner_labels[:,i].ravel()
            res = ClassificationResult(truth, classes, prbs_subset)
            name = key
            if inner_prbs.shape[1] > 1:
                # this is a hack. The data is stored with input name prefix,
                # but the key is PREFIX.output, so we replace output with 
                # the prefix. Note that we also assume htat TFnames is 
                # correctly set
                name += "_%s" % model.data[
                    key.replace('chipseq_output', '').replace('output', '')
                ].factor_names[i]
            val_results[name] = res
    return val_results

class MonitorAccuracy(keras.callbacks.Callback):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        super(MonitorAccuracy, self).__init__(*args, **kwargs)

    def _calc_val_results(self):
        return calc_val_results(self.model, self.batch_size)

    def on_epoch_end(self, batch_num, logs):
        val_losses = []
        for i, data in enumerate(self.model.iter_validation_data(
                self.batch_size, repeat_forever=False)):
            val_losses.append( self.model.test_on_batch(data) )
        val_loss = np.array(val_losses).mean()
        logs['val_loss'] = val_loss
        print "val_loss:", val_loss
            
        # calculate and print he validation results
        val_results = self._calc_val_results()

        # update the in-vitro penalty
        invitro_weight = self.model.invitro_weight.get_value()
        #print "Current Invitro Weight:",  invitro_weight
        if batch_num < 100:
            self.model.invitro_weight.set_value(
                np.array(1.0/np.log(2+batch_num), dtype='float32'))
            print "In-vitro weight is %.2f" % (
                self.model.invitro_weight.get_value())
        # print the validation results
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

    def _init_shared_convolution_layer(self):
        # initialize the full subdomain convolutional filter
        self.num_invivo_convs = 0
        self.num_tf_specific_invitro_binding_subdomain_convs = 1
        self.num_tf_specific_invivo_subdomain_convs = 0
        self.num_tf_specific_convs = (
            self.num_tf_specific_invitro_binding_subdomain_convs
            + self.num_tf_specific_invivo_subdomain_convs
        )
        self.num_affinity_outputs = (
            self.num_tf_specific_convs*len(self.factor_names)
            + self.num_invivo_convs
        )
        
        self.binding_subdomain_conv_size = 32
        self.stacked_subdomains_layer_shape = (
            self.num_affinity_outputs,
            3, 
            1,
            self.binding_subdomain_conv_size
        ) 
        self.stacked_subdomains_layer_filters = (
            self.stacked_subdomains_layer_shape,
            K.zeros((self.stacked_subdomains_layer_shape[0],)),
            initializations.get('glorot_uniform')(
                self.stacked_subdomains_layer_shape)
        )
        # extract the tf specific subtensors
        self.binding_subdomains_layer = ConvolutionDNASequenceBinding(
            self.num_affinity_outputs,
            self.binding_subdomain_conv_size, # convWidth,
        )
        self.binding_subdomains_layer.W_shape = (
            self.stacked_subdomains_layer_filters[0] )
        self.binding_subdomains_layer.b = (
            self.stacked_subdomains_layer_filters[1], None)
        self.binding_subdomains_layer.W = (
            self.stacked_subdomains_layer_filters[2], None)

        self.binding_subdomain_layers = {}
        step_size = ( 
            self.num_tf_specific_invitro_binding_subdomain_convs 
            + self.num_tf_specific_invivo_subdomain_convs
        )
        for i, factor_name in enumerate(self.factor_names):
            subtensor_slice = slice(
                i*step_size,
                i*step_size+self.num_tf_specific_invitro_binding_subdomain_convs
            )
            W_shape = list(self.stacked_subdomains_layer_shape)
            W_shape[0] = self.num_tf_specific_invitro_binding_subdomain_convs
            W_shape = tuple(W_shape)
            b = self.stacked_subdomains_layer_filters[1] #subtensor_slice]
            W = self.stacked_subdomains_layer_filters[2] #[subtensor_slice]
            binding_sub_domains_layer = ConvolutionDNASequenceBinding(
                self.num_tf_specific_invitro_binding_subdomain_convs, # numConv
                self.binding_subdomain_conv_size, # convWidth,
            )
            binding_sub_domains_layer.W_shape = W_shape
            binding_sub_domains_layer.W = (W, subtensor_slice)
            binding_sub_domains_layer.b = (b, subtensor_slice)
            self.binding_subdomain_layers[factor_name] = binding_sub_domains_layer

        return

    def __init__(self, num_samples, factor_names, validation_split=0.1, *args, **kwargs):
        # store a dictionary of round indexed selex sequences, 
        # indexed by experiment ID
        self.num_samples = num_samples
        assert self.num_samples%2 == 0
        self.validation_split = validation_split
        self.factor_names = factor_names

        self.data = {}
        self.named_losses = {}

        self.invitro_weight = K.variable(1.0)
        
        # initialize the base convolutional filter
        self.num_small_convs = 64
        self.small_conv_size = 4
        self.short_conv_layer = ConvolutionDNASequenceBinding(
            self.num_small_convs, # numConv
            self.small_conv_size, # convWidth, 
            #init=initial_model
        )
        self.short_conv_layer.init_filters()

        self._init_shared_convolution_layer()

        super(Graph, self).__init__(*args, **kwargs)

    def add_affinity_layers(self, input_name, output_name, factor_name=None):
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
        if factor_name is None:
            self.add_node(
                self.binding_subdomains_layer.create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        else:
            assert factor_name in self.binding_subdomain_layers, factor_name
            self.add_node(
                self.binding_subdomain_layers[factor_name].create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        return
    
    
    def add_occupancy_layers(self, input_name, output_name):
        self.add_node(
            LogNormalizedOccupancy(-10.0, None),
            name=output_name,
            input=input_name
        )

        #self.add_input(
        #    name=name_prefix+'accessibility_data', 
        #    input_shape=(1, 977, 1)) # seq_length        
        #self.add_node(
        #    Convolution2D(4, 16, 32),
        #    name=name_prefix+'occupancies.2',
        #    input=name_prefix+'occupancies.1'
        #)
        #        name_prefix+'occupancies.1',
        #        name_prefix+'accessibility_data'
        #    ]
        #)
        # find the log any bound occupancy for every track
        
        return
    
    def add_chipseq_regularization(
            self, occupancies_input_name, name_prefix, output_dim):
        self.add_node(
            LogAnyBoundOcc(),
            name=name_prefix+'csr.'+'max.1', 
            input=occupancies_input_name)
        self.add_node(
            OccMaxPool(2*self.num_tf_specific_convs, 'full'),
            name=name_prefix+'csr.'+'max.2',
            input=name_prefix+'csr.'+'max.1'
        )
        self.add_node(
            Flatten(),
            name=name_prefix+'csr.'+'max.3',
            input=name_prefix+'csr.'+'max.2'
        )
        self.add_node(
            Lambda(lambda x: K.exp(x)),
            name=name_prefix+'csr.'+'max.4',
            input=name_prefix+'csr.'+'max.3'
        )
        self.add_output(
            name=name_prefix+'chipseq_output', 
            input=name_prefix+'csr.'+'max.4')
        self.named_losses[name_prefix + 'chipseq_output'] = weighted_loss_generator(
            1.0, mse_skip_ambig) 
        return
    
    def add_invivo_layer(self, name_prefix, seq_length, output_dim):
        print "add_invivo_layer", name_prefix, seq_length, output_dim
        self.add_input(
            name=name_prefix+'fwd_seqs', 
            input_shape=(4, 1, seq_length))
        self.add_affinity_layers(
            input_name=name_prefix+'fwd_seqs', 
            output_name=name_prefix+'binding_affinities.1')
        #self.add_node(
        #    ConvolutionBindingSubDomains(
        #        nb_domains=self.num_affinity_outputs, 
        #        domain_len=32,
        #    ),
        #    name=name_prefix + 'binding_affinities.2',
        #    input=name_prefix+'binding_affinities.1'
        #)
        self.add_occupancy_layers(
            input_name=name_prefix+'binding_affinities.1', 
            output_name=name_prefix+'occupancies.1'
        )

        ###################################################################
        self.add_chipseq_regularization(
            name_prefix+'occupancies.1', name_prefix, output_dim)

        ###################################################################
        if True:
            #self.add_node(
            #    OccMaxPool(1, 8), #961),
            #    name=name_prefix +'max.1',
            #    input=name_prefix +'occupancies.1'
            #)
            #self.add_node(
            #    ConvolutionCoOccupancy(
            #        2, 1
            #        self.num_affinity_outputs, 32
            #    ),
            #    name=name_prefix + 'max.2',
            #    input=name_prefix+'occupancies.1'
            #)
            self.add_node(
                OccMaxPool(1, 64), #961),
                name=name_prefix +'max.4',
                input=name_prefix +'occupancies.1'
            )
            self.add_node(
                Lambda(lambda x: K.exp(x)),
                name=name_prefix +'max.8',
                input=name_prefix +'max.4'
            )
            self.add_node(
                Flatten(),
                name=name_prefix +'max.9',
                input=name_prefix +'max.8'
            )
            self.add_node(
                Dense(output_dim=output_dim),
                name=name_prefix +'dense',
                input=name_prefix +'max.9'
            )
            self.add_output(
                name=name_prefix+'output', 
                input=name_prefix +'dense')
            self.named_losses[name_prefix + 'output'] = weighted_loss_generator(
                1.0, mse_skip_ambig) 
        
        return
    
    def add_chipseq_samples(self, pks_and_labels): 
        print "Adding ChIP-seq data for sample ID %s" % pks_and_labels.sample_id
        name_prefix = 'invivo_%s_sequence.' % pks_and_labels.sample_id 
        self.data[name_prefix] = pks_and_labels
        self.add_invivo_layer(
            name_prefix, 
            pks_and_labels.fwd_seqs.shape[3], 
            pks_and_labels.labels.shape[1])
        return

    def add_invitro_layer(self, name_prefix, factor_name, seq_length):
        print "add_invitro_layer", name_prefix, seq_length
        self.add_input(
            name=name_prefix+'fwd_seqs', 
            input_shape=(4, 1, seq_length))

        self.add_affinity_layers(
            input_name=name_prefix+'fwd_seqs', 
            output_name=name_prefix + 'binding_affinities',
            factor_name=factor_name)
        #N X 2 X seq_len X num_filters

        self.add_node(
            LogNormalizedOccupancy(0.0),
            name=name_prefix + 'occupancies',
            input=name_prefix + 'binding_affinities')
        #N X 1 X seq_len X 2*num_filters

        self.add_node(
            LogAnyBoundOcc(),
            name=name_prefix + 'max.1',
            input=name_prefix + 'occupancies')

        self.add_node(
            Lambda(lambda x: K.exp(
                K.max(K.batch_flatten(x), axis=1, keepdims=True))),
            name=name_prefix+'max.2', 
            input=name_prefix+'max.1')
        #N X 1

        self.add_output(name=name_prefix+'output', input=name_prefix+'max.2')

        return

    def add_selex_experiments(self, selex_experiments):
        for exp in selex_experiments:
            self.add_selex_experiment(exp, weight=1.0)

    def add_selex_experiment(self, selex_experiment, weight=1.0):
        selex_exp_id = selex_experiment.selex_exp_id
        factor_name = selex_experiment.factor_name
        factor_id = selex_experiment.factor_id
        seq_length = selex_experiment.fwd_seqs.shape[3]
        
        name_prefix = 'invitro_%s_%s_%s.' % (
            factor_name, factor_id, selex_exp_id)
        if VERBOSE: print "Adding", name_prefix
        if factor_name not in self.binding_subdomain_layers:
            return
        self.data[name_prefix] = selex_experiment        
        self.add_invitro_layer(name_prefix, factor_name, seq_length)
        self.named_losses[name_prefix + 'output'] = weighted_loss_generator(
            self.invitro_weight*weight, mse_skip_ambig)

    def compile(self, *args, **kwargs):
        #sample_weight_modes = {}
        #for key in self.named_losses.keys():
        #    if key.startswith('invivo'): 
        #        sample_weight_modes[key] = None
        
        compile = functools.partial(
            super(JointBindingModel, self).compile,
            loss=self.named_losses,
            optimizer=Adam(),
            #sample_weight_modes=sample_weight_modes
        )
        return compile(*args, **kwargs)

    def iter_batches(
            self, batch_size, data_subset, repeat_forever, oversample=True):
        assert data_subset in ('train', 'validation')
        # initialize the set of iterators
        iterators = {}
        for key, data in self.data.iteritems():
            iterators[key] = data.iter_batches(
                batch_size, data_subset, repeat_forever)

        # decide how much to oversample
        if oversample is True and data_subset == 'train':
            num_oversamples = 5
        else:
            num_oversamples = 1

        def iter_data():
            while True:
                merged_output = {}
                for name_prefix, iterator in iterators.iteritems():
                    data_dict = next(iterator)
                    for key, data in data_dict.iteritems():
                        assert name_prefix + key not in merged_output
                        merged_output[name_prefix + key] = data
                yield merged_output
        
        return iter_weighted_batch_samples(self, iter_data(), num_oversamples)
    
    def iter_train_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'train', repeat_forever)
    
    def iter_validation_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'validation', repeat_forever)

    def fit(self, 
            validation_split=0.1, 
            batch_size=100, 
            nb_epoch=100,
            *args, **kwargs):
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto')
        monitor_accuracy_cb = MonitorAccuracy(batch_size)

        fit = functools.partial(
            super(JointBindingModel, self).fit_generator,
            self.iter_train_data(batch_size, repeat_forever=True),
            samples_per_epoch=int(self.num_samples*(1-self.validation_split)),
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[monitor_accuracy_cb,], # early_stop],
        )
        return fit(*args, **kwargs)

class SamplePeaksAndLabels():
    def one_hot_code_peaks_sequence(self):
        cached_fname = "cachedseqs.%s.obj" % hashlib.sha1(self.pks.view(np.uint8)).hexdigest()
        try:
            with open(cached_fname) as fp:
                print "Loading cached seqs"
                return np.load(fp)
        except IOError:
            pass

        pk_width = self.pks[0][2] - self.pks[0][1]
        rv = 0.25 * np.ones((len(self.pks), pk_width, 4), dtype='float32')
        for i, data in enumerate(self.pks):
            assert pk_width == data[2] - data[1]
            seq = self.genome_fasta.fetch(str(data[0]), data[1], data[2])
            if len(seq) != pk_width: continue
            coded_seq = one_hot_encode_sequence(seq)
            rv[i,:,:] = coded_seq
        
        # add the extra dimension for theano
        rv = np.swapaxes(rv, 1, 2)[:,:,None,:]
        
        with open(cached_fname, "w") as ofp:
            print "Saving seqs"
            np.save(ofp, rv)

        return rv

    def load_accessibility_data(self):
        from pyTFbindtools.DB import load_dnase_fnames
        # get the correct filename
        fnames = load_dnase_fnames([self.sample_id,])

        cached_fname = "cachedaccessibility.%s.%s.obj" % (
            hashlib.sha1(self.pks.view(np.uint8)).hexdigest(),
            hash(tuple(fnames))
        )
        try:
            with open(cached_fname) as fp:
                print "Loading cached accessibility data"
                rv = np.load(fp)
        except IOError:
            rv = bigWigFeaturize.new(
                fnames,
                self.half_peak_width*2, 
                intervals=[
                    Region(pk['contig'], pk['start'], pk['stop']) for pk in self.pks
                ]
            )

            with open(cached_fname, "w") as ofp:
                print "Saving accessibility data"
                np.save(ofp, rv)
        
        return rv

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
        all_factor_names = load_tf_names(self.tf_ids)
        # make sure all of the tfnames actually exist
        #assert ( self.desired_factor_names is None 
        #         or all(factor_name in all_factor_names 
        #                for factor_name in self.desired_factor_names) 
        #)
        # filter the tf set 
        filtered_tfs = sorted([
            (factor_name, tf_id, i) for i, (factor_name, tf_id) 
            in enumerate(zip(all_factor_names, self.tf_ids)) 
            if (self.desired_factor_names is None 
                or factor_name in self.desired_factor_names)
        ])
        self.factor_names, self.tf_ids, tf_indices = zip(*filtered_tfs)
        self.pks = self.pks[:self.n_samples]
        self.idr_optimal_labels = self.idr_optimal_labels[
            :self.n_samples, np.array(tf_indices)]
        self.relaxed_labels = self.relaxed_labels[
            :self.n_samples, np.array(tf_indices)]
        
        # set the ambiguous labels to -1
        self.ambiguous_pks_mask = (
            self.idr_optimal_labels != self.relaxed_labels)
        self.clean_labels = self.idr_optimal_labels.copy()
        self.clean_labels[self.ambiguous_pks_mask] = -1

        # print out balance statistics
        #print (self.idr_optimal_labels.sum(axis=0)/self.idr_optimal_labels.shape[0])
        #print (self.relaxed_labels.sum(axis=0)/self.relaxed_labels.shape[0])
        #print (self.ambiguous_pk_indices.sum(axis=0)/float(self.ambiguous_pk_indices.shape[0]))

        return
    
    def __init__(
            self, sample_id, n_samples, annotation_id=1, 
            half_peak_width=500, factor_names=None):
        self.sample_id = sample_id
        self.annotation_id = annotation_id
        self.half_peak_width = half_peak_width
        self.n_samples = n_samples
        self.desired_factor_names = factor_names
        self._init_pks_and_label_mats()


        #res = bigWigFeaturize.new(
        #    ["/mnt/data/epigenomeRoadmap/signal/consolidated/macs2signal/foldChange/E123-DNase.fc.signal.bigwig",], 
        #    self., 
        #    intervals=regions)

        # initialize the training and validation data
        index = np.argsort(np.random.random(len(self.pks)))
        self.pks = self.pks[index]
        self.idr_optimal_labels = self.idr_optimal_labels[index,:]
        self.relaxed_labels = self.relaxed_labels[index,:]
        self.ambiguous_labels = self.clean_labels[index,:]

        training_index = int(self.n_samples*0.1)        
        self.labels = self.ambiguous_labels # idr_optimal_labels
        self.train_labels = self.labels[training_index:]
        self.validation_labels = self.labels[:training_index]
        
        # code the peaks' sequence
        print "Coding peaks"
        self.genome_fasta = FastaFile(
            load_genome_metadata(self.annotation_id).filename)
        self.fwd_seqs = self.one_hot_code_peaks_sequence()
        print self.fwd_seqs.shape
        print "="*40
        self.train_fwd_seqs = self.fwd_seqs[training_index:]
        self.validation_fwd_seqs = self.fwd_seqs[:training_index]

        print "Loading Accessibility Data"
        print len(self.pks)
        self.accessibility_data = self.load_accessibility_data()[:,:,:977,:]
        print self.accessibility_data.shape
        assert self.accessibility_data.shape[0] == n_samples
        print "="*40
        self.train_accessibility_data = self.accessibility_data[training_index:]
        self.validation_accessibility_data = self.accessibility_data[
            :training_index]

    def iter_batches(self, batch_size, data_subset, repeat_forever):
        if data_subset == 'train':
            accessibility_data = self.train_accessibility_data
            fwd_seqs = self.train_fwd_seqs
            labels = self.train_labels
        elif data_subset == 'validation':
            accessibility_data = self.validation_accessibility_data
            fwd_seqs = self.validation_fwd_seqs
            labels = self.validation_labels
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        i = 0
        n = fwd_seqs.shape[0]//batch_size
        if n <= 0: raise ValueError, "Maximum batch size is %i (requested %i)" \
           % (fwd_seqs.shape[0], batch_size)
        permutation = None
        while repeat_forever is True or i < n:
            if i%n == 0:
                permutation = np.random.permutation(labels.shape[0])
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            indices = permutation[subset]
            yield {'fwd_seqs': fwd_seqs[indices], 
                   'accessibility_data': accessibility_data[indices], 
                   'chipseq_output': labels[indices],
                   'output': labels[indices]
            }
            i += 1
        
        return
    
    def iter_train_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'train', repeat_forever)

    def iter_validation_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'validation', repeat_forever)

class SelexExperiment():
    @property
    def seq_length(self):
        return self.fwd_seqs.shape[3]

    def get_coded_seqs_and_labels(self, unbnd_seqs, bnd_seqs, primer):
        left_flank, right_flank = re.split('[0-9]{1,4}N', primer)
        left_flank = left_flank.rjust(self.seq_pad).replace(' ', 'N')
        right_flank = right_flank.ljust(self.seq_pad).replace(' ', 'N')

        num_seqs = min(len(unbnd_seqs), len(bnd_seqs))
        unbnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in unbnd_seqs[:num_seqs]], 
            include_shape=False)
        bnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in bnd_seqs[:num_seqs]], 
            include_shape=False)

        permutation = np.random.permutation(2*num_seqs)
        fwd_one_hot_seqs = np.vstack(
            (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs)
        )[permutation,:,:]
        fwd_one_hot_seqs = np.swapaxes(fwd_one_hot_seqs, 1, 2)[:,:,None,:]

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

    def __init__(self, selex_exp_id, n_samples, validation_split=0.1, seq_pad=40):
        self.selex_exp_id = selex_exp_id
        self.n_samples = n_samples
        self.seq_pad = seq_pad
        self.validation_split = validation_split
        self._training_index = int(self.n_samples*self.validation_split)

        # load connect to the DB, and find the factor name
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', selex_exp_id)
        self.factor_name = selex_db_conn.get_factor_name()
        self.factor_id = selex_db_conn.get_factor_id()

        # load the sequencess
        if VERBOSE:
            print "Loading sequences for %s-%s (exp ID %i)" % (
                self.factor_name, self.factor_id, self.selex_exp_id)
        self.primer, fnames, bg_fname = selex_db_conn.get_primer_and_fnames()
        if bg_fname is None:
            raise NoBackgroundSequences(
                "No background sequences for %s (%i)." % (
                    self.factor_name, self.selex_exp_id))

        with optional_gzip_open(fnames[max(fnames.keys())]) as fp:
            bnd_seqs = load_fastq(fp, self.n_samples/2)
        with optional_gzip_open(bg_fname) as fp:
            unbnd_seqs = load_fastq(fp, self.n_samples/2)
        if len(bnd_seqs[0]) < 20:
            raise SeqsTooShort("Seqs too short for %s (exp %i)." % (
                self.factor_name, selex_exp_id))
        
        if len(unbnd_seqs) < self.n_samples/2:
            unbnd_seqs = upsample(unbnd_seqs, self.n_samples/2)
        if len(bnd_seqs) < self.n_samples/2:
            bnd_seqs = upsample(bnd_seqs, self.n_samples/2)

        (self.fwd_seqs, self.shape_seqs, self.labels 
             ) = self.get_coded_seqs_and_labels(
                 unbnd_seqs, bnd_seqs, self.primer)

        self.train_fwd_seqs = self.fwd_seqs[self._training_index:]
        self.train_labels = self.labels[self._training_index:]
        
        self.validation_fwd_seqs = self.fwd_seqs[:self._training_index]
        self.validation_labels = self.labels[:self._training_index]
        
        return

    def iter_batches(self, batch_size, data_subset, repeat_forever):
        if data_subset == 'train': 
            fwd_seqs = self.train_fwd_seqs
            labels = self.train_labels
        elif data_subset == 'validation':
            fwd_seqs = self.validation_fwd_seqs
            labels = self.validation_labels
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        i = 0
        n = fwd_seqs.shape[0]//batch_size
        if n <= 0: raise ValueError, "Maximum batch size is %i (requested %i)" \
           % (fwd_seqs.shape[0], batch_size)
        while repeat_forever is True or i < n:
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            yield {'fwd_seqs': fwd_seqs[subset], 'output': labels[subset,None]}
            i += 1
        
        return
    
    def iter_train_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'train', repeat_forever)

    def iter_validation_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'validation', repeat_forever)

class SelexData():
    def load_tfs_grpd_by_family(self):
        raise NotImplementedError, "This fucntion just exists to save teh sql query"
        query = """
          SELECT tfs.family_id, family_name, array_agg(tfs.factor_name) 
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN tf_families 
        GROUP BY family_id, family_name;
        """ # group tfs by their families

        pass

    def find_all_selex_experiments(self):
        import psycopg2
        conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % ('mitra', 'cisbp', 'nboley'))
        cur = conn.cursor()    
        query = """
          SELECT selex_experiments.tf_id, tfs.tf_name, array_agg(distinct selex_exp_id)
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN roadmap_matched_chipseq_peaks
        GROUP BY selex_experiments.tf_id, tfs.tf_name
        ORDER BY tf_name;
        """
        cur.execute(query)
        # filter out the TFs that dont have background sequence or arent long enough
        res = [ x for x in cur.fetchall() 
                if x[1] not in ('E2F1', 'E2F4', 'POU2F2', 'PRDM1', 'RXRA')
                #and x[1] in ('MAX',) # 'YY1', 'RFX5', 'USF', 'PU1', 'CTCF')
        ]
        return res

    def add_selex_experiment(self, selex_exp_id):
        self.experiments.append(
            SelexExperiment(
                selex_exp_id, self.n_samples, self.validation_split)
        )

    def add_all_selex_experiments_for_factor(self, factor_name):
        exps_added = 0
        for tf_id, i_factor_name, selex_exp_ids in self.find_all_selex_experiments():
            if i_factor_name == factor_name:
                for exp_id in selex_exp_ids:
                    self.add_selex_experiment(exp_id)
                    exps_added += 1
        if exps_added == 0:
            print "Couldnt add for", factor_name
        #assert exps_added > 0
        return

    def add_all_selex_experiments(self):
        for tf_id, factor_name, selex_exp_ids in self.find_all_selex_experiments():
            for selex_exp_id in selex_exp_ids:
                try: 
                    self.add_selex_experiment(selex_exp_id)
                except SeqsTooShort, inst:
                    print inst
                    continue
                except NoBackgroundSequences, inst:
                    print inst
                    continue
                except Exception, inst:
                    raise
    
    @property
    def factor_names(self):
        return [exp.factor_name for exp in self]

    def __iter__(self):
        for experiment in self.experiments:
            yield experiment
        return

    def __init__(self, n_samples, validation_split=0.1):
        self.n_samples = n_samples
        self.validation_split = 0.1
        self.experiments = []

    def __len__(self):
        return len(self.experiments)


def old_main():
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

    model.add_selex_layer(441) # MAX
    model.add_selex_layer(202) # YY1
    model.add_chipseq_layer('T011266_1.02', 'MAX') 
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=5)

    pass

def fit_selex(n_samples):
    selex_experiments = SelexData(n_samples)
    for tf_id, tf_name, exp_ids in selex_experiments.find_all_selex_experiments():
        if tf_name != 'CTCF': continue
        for i, exp_id in enumerate(exp_ids):
            exp = SelexExperiment(exp_id, n_samples)
            model = JointBindingModel(n_samples, [exp.factor_name,])
            model.add_selex_experiment(exp)
            model.compile()
            model.fit(validation_split=0.1, batch_size=100, nb_epoch=15)
            print tf_name, i+1, calc_val_results(model, 100).values()[0]
    return

from pyTFbindtools.peaks import build_peaks_label_mat
def main():
    n_samples = 50000 #100000 # 50000 # 300000
    sample_ids = ['E123',] # 'E119']
    factor_names = [
        'BHLHE40',  'CEBPB', 'CTCF', 'ELF1',  'ELK1', 'ETS1', 'MAX', #'MAX', #'NANOG'# 'ESRRA', 
        #'SP1', 'HSF1', 'TCF3', 'GATA1', 'RELA', 'IRF1', 
    ] 
    #    'HSF1', 'SP1', 'TCF3', 'GATA1', 'RELA', 'POLR2AphosphoS2', 'IRF1', 
    #    'JUN', 'JUND', 'KAT2B', 'FOS', 'SPI1', 'USF2']
    #factor_names = [
    #    'CTCF', 'MAX', 'TCF12', 'MYC', 'YY1', 'REST', 'TCF21', 'TCF4', 'TCF7']
    factor_names = ['ELK1', 'BHLHE40', 'CTCF', 'MAX'] #['TBP', 'CTCF', 'YY1', 'MAX', 'TCF21']
    factor_names = ['CTCF',] #['TBP', 'CTCF', 'YY1', 'MAX', 'TCF21']
    #factor_names=None
    all_sample_peaks_and_labels = [
        SamplePeaksAndLabels(
            sample_id, n_samples, factor_names=factor_names)
        for sample_id in sample_ids
    ]
    selex_experiments = SelexData(n_samples)
    #selex_experiments.add_all_selex_experiments()
    for factor_name in factor_names:
        selex_experiments.add_all_selex_experiments_for_factor(factor_name)
    #all_factor_names = list(set(
    #    list(all_sample_peaks_and_labels[0].factor_names) 
    #    + list(selex_experiments.factor_names)
    #))
    #print sorted(all_factor_names)
    all_factor_names = factor_names
    model = JointBindingModel(n_samples, all_factor_names)
    for sample_peaks_and_labels in all_sample_peaks_and_labels:
        model.add_chipseq_samples(sample_peaks_and_labels)
    model.add_selex_experiments(selex_experiments)
    
    print "Compiling Model"
    model.compile()
    model.fit(validation_split=0.1, batch_size=100, nb_epoch=50)

    for i, mo in enumerate(
            model.binding_subdomains_layer.extract_binding_models()):
        mo.build_pwm_model(-6.0).plot("conv.%i.png" % i)
    return

    model = JointBindingModel(n_samples, factor_names)
    
    model.add_chipseq_layer('T011266_1.02', 'MAX') 
    #model.add_chipseq_layer('T044261_1.02', 'YY1')

    #for tf_id, factor_name, selex_exp_ids in res:
    #    try:
    #        model.add_chipseq_layer(tf_id, factor_name)
    #    except Exception, inst:
    #        print "Couldnt load ChIP-seq data for %s" % factor_name
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
