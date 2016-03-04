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

from pyTFbindtools.peaks import SelexData, SelexExperiment, SamplePeaksAndLabels
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

def zero_loss(y_true, y_pred):
    return np.zeros(1)

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
    return K.mean(
        K.square(y_pred*non_ambig - y_true*non_ambig)*y_true.shape[0]/cnts, 
        axis=-1
    ) 

def weighted_loss_generator(weight, loss_fn):
    def f(y_true, y_pred):
        return weight*loss_fn(y_true, y_pred)
    return f

def calc_val_results(model, batch_size):
    data_iterator = model.iter_validation_data(
        batch_size, repeat_forever=False)
    pred_prbs = defaultdict(list)
    labels = defaultdict(list)
    pred_prbs = []
    labels = []
    for i, (data, labels) in enumerate(data_iterator):
        prbs = model.predict_on_batch(data)
        pred_prbs.append(prbs)
        labels.append(labels)
    pred_prbs = np.vstack(pred_prbs)
    labels = np.vstack(labels)
    return pred_prbs, labels
    
        for key, prbs in model.predict_on_batch(data).iteritems():
            print key
            if key.endswith('.accessibility_data'): continue
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
    def __init__(self, data, batch_size, *args, **kwargs):
        self.data = data
        self.batch_size = batch_size
        super(MonitorAccuracy, self).__init__(*args, **kwargs)

    def _calc_val_results(self):
        return calc_val_results(self.data, self.batch_size)

    def on_epoch_end(self, batch_num, logs):
        val_losses = []
        for i, data in enumerate(self.data.iter_validation_data(
                self.batch_size, repeat_forever=False)):
            val_losses.append( self.model.test_on_batch(data) )
        val_loss = np.array(val_losses).mean()
        logs['val_loss'] = val_loss
            
        # calculate and print he validation results
        val_results = self._calc_val_results()

        # update the in-vitro penalty
        invitro_weight = self.data.invitro_weight.get_value()
        #print "Current Invitro Weight:",  invitro_weight
        if batch_num < 100:
            self.data.invitro_weight.set_value(
                np.array(1.0/np.log(2+batch_num), dtype='float32'))
            print "In-vitro weight is %.2f" % (
                self.data.invitro_weight.get_value())
        # print the validation results
        for key, res in sorted(val_results.iteritems()):
            print key.ljust(40), res
        

def calc_accuracy(pred_probs, labels):
    return float((pred_probs.round() == labels[:,None]).sum())/labels.shape[0]

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
        self.num_tf_specific_invitro_binding_subdomain_convs = 4
        self.num_tf_specific_invivo_subdomain_convs = 4
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

    def __init__(self, factor_names, validation_split=0.1, *args, **kwargs):
        # store a dictionary of round indexed selex sequences, 
        # indexed by experiment ID
        self.validation_split = validation_split
        self.factor_names = factor_names

        self.data = {}
        self.named_losses = {}
        self.occupancy_layers = []

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

        self.model = Graph()
        #super(Graph, self).__init__(*args, **kwargs)
        
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
        
        self.model.add_node(
            short_conv_layer,
            input=input_name,
            name=input_name + ".1"
        )
        """

        subdomain_input_name = input_name # + ".1"
        if factor_name is None:
            self.model.add_node(
                self.binding_subdomains_layer.create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        else:
            assert factor_name in self.binding_subdomain_layers, factor_name
            self.model.add_node(
                self.binding_subdomain_layers[factor_name].create_clone(),
                input=subdomain_input_name,
                name=output_name
            )
        return
    
    
    def add_occupancy_layers(self, name_prefix, input_name, output_name):
        self.model.add_node(
            LogNormalizedOccupancy(-10.0, None),
            name=output_name,
            input=input_name
        )

        #self.model.add_output(
        #    name=name_prefix+'binding_occupancies', 
        #    input=output_name)
        #self.named_losses[name_prefix+'binding_occupancies'] = zero_loss
        #
        #self.occupancy_layers.append(name_prefix+'binding_occupancies')
        #self.model.add_input(
        #    name=name_prefix+'accessibility_data', 
        #    input_shape=(1, 977, 1)) # seq_length        
        #self.model.add_node(
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
    
    def predict_occupancies(self, data_iter=None):
        rv = defaultdict(list)
        if data_iter is None:
            data_iter = self.iter_train_data(100, repeat_forever=False)
        for data in data_iter:
            output = self.predict_on_batch(data)
            for key in self.occupancy_layers:
                rv[key].append(output[key])
        for key, val in rv.iteritems():
            rv[key] = np.vstack(val)
        return dict(rv)
    
    def add_chipseq_regularization(
            self, occupancies_input_name, name_prefix, output_dim):
        self.model.add_node(
            LogAnyBoundOcc(),
            name=name_prefix+'csr.'+'max.1', 
            input=occupancies_input_name)
        self.model.add_node(
            OccMaxPool(2*self.num_tf_specific_convs, 'full'),
            name=name_prefix+'csr.'+'max.2',
            input=name_prefix+'csr.'+'max.1'
        )
        self.model.add_node(
            Flatten(),
            name=name_prefix+'csr.'+'max.3',
            input=name_prefix+'csr.'+'max.2'
        )
        self.model.add_node(
            Lambda(lambda x: K.exp(x)),
            name=name_prefix+'csr.'+'max.4',
            input=name_prefix+'csr.'+'max.3'
        )
        self.model.add_output(
            name=name_prefix+'chipseq_output', 
            input=name_prefix+'csr.'+'max.4')
        self.named_losses[name_prefix + 'chipseq_output'] = weighted_loss_generator(
            1.0, mse_skip_ambig) 
        return
    
    def add_invivo_layer(self, name_prefix, seq_length, output_dim):
        print "add_invivo_layer", name_prefix, seq_length, output_dim
        self.model.add_input(
            name=name_prefix+'fwd_seqs', 
            input_shape=(4, 1, seq_length))
        self.add_affinity_layers(
            input_name=name_prefix+'fwd_seqs', 
            output_name=name_prefix+'binding_affinities.1')
        #self.model.add_node(
        #    ConvolutionBindingSubDomains(
        #        nb_domains=self.num_affinity_outputs, 
        #        domain_len=32,
        #    ),
        #    name=name_prefix + 'binding_affinities.2',
        #    input=name_prefix+'binding_affinities.1'
        #)
        self.add_occupancy_layers(
            name_prefix,
            input_name=name_prefix+'binding_affinities.1', 
            output_name=name_prefix+'occupancies.1'
        )

        ###################################################################
        self.add_chipseq_regularization(
            name_prefix+'occupancies.1', name_prefix, output_dim)

        ###################################################################
        if False:
            #self.model.add_node(
            #    OccMaxPool(1, 8), #961),
            #    name=name_prefix +'max.1',
            #    input=name_prefix +'occupancies.1'
            #)
            #self.model.add_node(
            #    ConvolutionCoOccupancy(
            #        2, 1
            #        self.num_affinity_outputs, 32
            #    ),
            #    name=name_prefix + 'max.2',
            #    input=name_prefix+'occupancies.1'
            #)
            self.model.add_node(
                OccMaxPool(1, 64), #961),
                name=name_prefix +'max.4',
                input=name_prefix +'occupancies.1'
            )
            self.model.add_node(
                Lambda(lambda x: K.exp(x)),
                name=name_prefix +'max.8',
                input=name_prefix +'max.4'
            )
            self.model.add_node(
                Flatten(),
                name=name_prefix +'max.9',
                input=name_prefix +'max.8'
            )
            self.model.add_node(
                Dense(output_dim=output_dim),
                name=name_prefix +'dense',
                input=name_prefix +'max.9'
            )
            self.model.add_output(
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
        self.model.add_input(
            name=name_prefix+'fwd_seqs', 
            input_shape=(4, 1, seq_length))

        self.add_affinity_layers(
            input_name=name_prefix+'fwd_seqs', 
            output_name=name_prefix + 'binding_affinities',
            factor_name=factor_name)
        #N X 2 X seq_len X num_filters

        self.model.add_node(
            LogNormalizedOccupancy(0.0),
            name=name_prefix + 'occupancies',
            input=name_prefix + 'binding_affinities')
        #N X 1 X seq_len X 2*num_filters

        self.model.add_node(
            LogAnyBoundOcc(),
            name=name_prefix + 'max.1',
            input=name_prefix + 'occupancies')

        self.model.add_node(
            OccMaxPool('full', 'full'),
            name=name_prefix+'max.2', 
            input=name_prefix+'max.1')
        self.model.add_node(
            Flatten(),
            name=name_prefix+'max.3', 
            input=name_prefix+'max.2')
        self.model.add_node(
            Lambda(lambda x: K.exp(x)),
            name=name_prefix+'max.4', 
            input=name_prefix+'max.3')
        #N X 1
        self.model.add_output(
            name=name_prefix+'output', input=name_prefix+'max.4')

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
            self.model.compile,
            loss=self.named_losses,
            optimizer=Adam(),
            #sample_weight_modes=sample_weight_modes
        )
        return compile(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def test_on_batch(self, *args, **kwargs):
        return self.model.test_on_batch(*args, **kwargs)

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
            num_oversamples = 2
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
        
        return iter_weighted_batch_samples(
            self.model, iter_data(), num_oversamples)
    
    def iter_train_data(self, batch_size, repeat_forever=False, oversample=False):
        return self.iter_batches(batch_size, 'train', repeat_forever, oversample)
    
    def iter_validation_data(self, batch_size, repeat_forever=False):
        return self.iter_batches(batch_size, 'validation', repeat_forever=repeat_forever, oversample=False)

    def fit_generator(self, 
            samples_per_epoch,
            batch_size=100, 
            nb_epoch=100,
            *args, **kwargs):
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto')
        monitor_accuracy_cb = MonitorAccuracy(self, batch_size)

        fit = functools.partial(
            self.model.fit_generator,
            self.iter_train_data(batch_size, repeat_forever=True),
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[monitor_accuracy_cb,], # early_stop],
        )
        return fit(*args, **kwargs)

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

def build_model_from_factor_names_and_sample_ids(
        n_samples, factor_names, sample_ids, include_selex=True, include_invivo=True):
    model = JointBindingModel(factor_names)

    if include_invivo:
        for sample_id in sample_ids:
            model.add_chipseq_samples(
                SamplePeaksAndLabels(
                    sample_id, n_samples, factor_names=factor_names)
            )

    if include_selex:
        selex_experiments = SelexData(n_samples)
        for factor_name in factor_names:
            selex_experiments.add_all_selex_experiments_for_factor(factor_name)
        model.add_selex_experiments(selex_experiments)
    

    return model

def main():
    n_samples = 1000 #500000 #100000 # 50000 # 300000

    sample_ids = ['E117',] # 'E119']
    factor_names = [
        'BHLHE40',  'CEBPB', 'CTCF', 'ELF1',  'ELK1', 'ETS1', 'MAX', #'MAX', #'NANOG'# 'ESRRA', 
        #'SP1', 'HSF1', 'TCF3', 'GATA1', 'RELA', 'IRF1', 
    ] 
    #factor_names = [
    #    'CTCF', 'MAX', 'TCF12', 'MYC', 'YY1', 'REST', 'TCF21', 'TCF4', 'TCF7']
    factor_names = ['ELK1', 'BHLHE40', 'CTCF', 'MAX'] #['TBP', 'CTCF', 'YY1', 'MAX', 'TCF21']
    factor_names = ['CTCF', 'MAX'] #'YY1', 'MYC'] #['TBP', 'CTCF', 'YY1', 'MAX', 'TCF21']

    model = build_model_from_factor_names_and_sample_ids(
        n_samples, factor_names, sample_ids, 
        include_selex=True, include_invivo=False)

    print "Compiling Model"
    model.compile()
    model.fit_generator(100)
    #occs = model.predict_occupancies()
    #print occs.keys()
    #for x in occs.values():
    #    print "="*20, x.shape
    
    model.fit_generator(n_samples*0.9, batch_size=100, nb_epoch=1)
    model.model.save_weights('test.hdf5', overwrite=True)
    return

if __name__ == '__main__':    
    main()
