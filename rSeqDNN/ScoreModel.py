import os
import glob
import sys
import numpy as np
from scipy.stats import rankdata
from scipy.signal import correlate2d
from sklearn.metrics import roc_auc_score
from matplotlib import (
    pyplot as plt,
    image as mpimg )
from pyTFbindtools.peaks import getFileHandle
from plot import plot_bases

ENCODE_MOTIFS_PATH =  os.path.join(os.getcwd(),'encode_motifs.txt.gz')

def normalise_sequence_conv_weights(weights, bias, weights_height=4):
    """
    Normalization of sequence convolutions for scoring with deepLift.

    Parameters
    ----------
    weights : 4d array
        Expects single channeled weights.
    bias : 1d array
    weight_height : int, default: 4
        height dimension of the weights (axis=2).

    Returns
    -------
    renormalised_weights : 4d array
    new_bias : 1d array
    """
    assert len(weights.shape)==4
    assert weights.shape[1]==1
    assert weights.shape[2]==weights_height
    mean_weights_at_positions = np.mean(weights,axis=2)
    new_bias = bias + np.sum(np.sum(mean_weights_at_positions, axis=2), axis=1)
    mean_weights_at_positions = mean_weights_at_positions[:,:,None,:]
    renormalised_weights = weights - mean_weights_at_positions

    return renormalised_weights, new_bias

def score_convolutions(model, X, batch_size):
    """
    Score convolutions using deepLifft on maxpooling layer.

    Parameters
    ----------
    model : keras model
        Expects sequential model with 1st Convolution2D layer,
        2nd MaxPooling2D layer, and final activation layer.
        Note: assumes non-recurrent model and piece-wise linear
        activations, otherwise scoring is not valid.
    X : 4d array
        one hot encoded sequences.
    batch_size : int

    Returns
    -------
    convolution_scores : 3d array
        (N, K, L) array with scores on N sequences of K convolutions
        with L maxpool windows.
    """
    from keras.layers.convolutional import Convolution2D
    from keras.layers.core import Activation
    assert isinstance(model.layers[0], Convolution2D), \
    "filter scoring requires initial Convolution2D layer!"
    assert isinstance(model.layers[-1], Activation), \
    "filter scoring requires final activation layer!"
    scripts_dir = os.environ.get("ENHANCER_SCRIPTS_DIR")
    os.sys.path.insert(0, "%s/featureSelector/deepLIFFT/kerasBasedBackprop" % scripts_dir)
    from deepLIFTonGPU import ScoreTypes, Activations_enum, OutLayerInfo, getScoreFunc

    layers_to_score = [model.layers[1]]
    output_layers = [OutLayerInfo(outLayNoAct=model.layers[-2],
                                  activation=Activations_enum.sigmoid)]
    scoring_function = getScoreFunc(model, layers_to_score, output_layers,
                                    [ScoreTypes.deepLIFT])
    deepLifft_score_list = []
    for start in xrange(0, len(X), batch_size):
        if (start+batch_size)>len(X):
                end  = len(X)
        else:
                end = start + batch_size
        scores = scoring_function([X[start:end, :, :, :]])
        deepLifft_scores = scores['deepLIFT'][0][0][0]
        deepLifft_score_list.append(deepLifft_scores.squeeze())
    convolution_scores = np.concatenate(tuple(deepLifft_score_list))
    return convolution_scores

def rank_convolutions(convolution_scores, y_true, y_pred, rank_metric='auROC'):
    """
    Rank contribution of convolutions to classification.

    Parameters
    ----------
    convolution_scores : 3d array
        (N, K, L) array with scores on N sequences of K convolutions
        with L maxpool windows.
    y_true : 1d array
        true labels.
    y_pred : 1d array
        predicted labels.
    rank_metric: string, optional.
        Metric used to rank convolutions.
        Legal values: 'auROC' (default), 'sensitivity', 'specificity'
    Returns
    -------
    convolution_ranks : 1darray
        convolution_ranks[j] is the rank of convolution j.
    """
    ## sum scores across maxpool windows for ranking                                                                                                                                                                                                                                            
    sum_scores = np.sum(convolution_scores, axis=2)
    if rank_metric=='auROC':
        auroc_scores = [roc_auc_score(y_true, sum_score) for sum_score in sum_scores.T]
        return rankdata(np.array(auroc_scores))
    elif rank_metric=='sensitivity':
        true_positives = (y_pred*y_true)==1
        false_negatives = ((1-y_pred)*y_true)==1
        tp_scores = sum_scores[true_positives].sum(axis=0)
        fn_scores = sum_scores[false_negatives].sum(axis=0)
        sensitivity_scores = tp_scores - fn_scores
        return rankdata(sensitivity_scores)
    elif rank_metric=='specificity':
        true_negatives = ((1-y_pred)*(1-y_true))==1
        false_positives = (y_pred*(1-y_true))==1
        tn_scores = sum_scores[true_negatives].sum(axis=0)
        fp_scores = sum_scores[false_positives].sum(axis=0)
        specificity_scores = -(tn_scores - fp_scores)
        return rankdata(specificity_scores)
    else:
        raise ValueError('Invalid rank option!')

class PWM(object):
    def __init__(self, weights, name=None, threshold=None):
        self.weights = weights
        self.name = name
        self.threshold = threshold

    @staticmethod
    def from_homer_motif(motif_file):
        with getFileHandle(motif_file) as fp:
            header = fp.readline().strip().split('\t')
            name = header[1]
            threshold = float(header[2])
            weights = np.loadtxt(fp)

        return PWM(weights, name, threshold)

    @staticmethod
    def get_encode_pwms(motif_file):
        pwms = []
        with getFileHandle(motif_file) as fp:
            line = fp.readline().strip()
            while True:
                if line == '':
                    break

                header = line
                weights = []
                while True:
                    line = fp.readline()
                    if line == '' or line[0] == '>':
                        break
                    weights.append(map(float, line.split()[1:]))
                pwms.append(PWM(np.array(weights), header))

        return pwms

    @staticmethod
    def from_cisbp_motif(motif_file):
        name = os.path.basename(motif_file)
        with getFileHandle(motif_file) as fp:
            _ = fp.readline()
            weights = np.loadtxt(fp)[:, 1:]
        return PWM(weights, name)

def max_min_cross_corr(pwm, conv_filter):
    if conv_filter.shape[1] != 4:
        conv_filter = conv_filter.T
    assert conv_filter.shape[1] == 4
    assert pwm.shape[1] == 4
    corr = correlate2d(pwm, conv_filter, mode='same')
    # we are only interested in the part where the 'letter' axes are aligned,
    # and the correlation is over the position axis only, which is in the 2nd
    # column
    allowed_corr = corr[:, 1]
    max_corr = np.max(allowed_corr)
    min_corr = np.min(allowed_corr)
    # max_pos (and min_pos) relates to the alignment between the pwm and the
    # conv_filter as follows - more generally,
    #
    # Position floor(w / 2) - i maps to position 0 on the (padded) PWM
    #
    # where w = width of conv filter, and i is the position given by best_pos
    # (or more generally, the index into allowed_corr). If a negative position
    # is obtained, that means that position 0 on the conv_filter maps to a
    # later position on the PWM.
    max_pos = np.argmax(allowed_corr)
    min_pos = np.argmin(allowed_corr)

    return ((max_corr, max_pos), (min_corr, min_pos))

def get_motif_matches(filters, pwms, topk=5):
    filterhits = []
    for conv_filter in filters:                
        conv_filter = conv_filter.squeeze()
        min_activation = conv_filter.min(axis=0).sum()
        max_activation = conv_filter.max(axis=0).sum()
        activation_range = max_activation - min_activation
        def norm_cc(cc):
            return (cc - min_activation) / activation_range
        hits = []
        for idx, pwm in enumerate(pwms):
            (max_cc, max_pos), (min_cc, min_pos) = \
                max_min_cross_corr(pwm.weights, conv_filter)
            hits.append((norm_cc(max_cc), max_pos, pwm.name))
            hits.append((norm_cc(min_cc), min_pos, pwm.name))
        hits.sort(reverse=True)
        filterhits.append(hits[:topk])

    return filterhits

def get_encode_pwm_hits(model):
    conv_weights, _ = model.layers[0].get_weights()
    encode_pwms = PWM.get_encode_pwms(ENCODE_MOTIFS_PATH)

    return get_motif_matches(conv_weights, encode_pwms)

def plot_convolutions(model, ofname, rank_dictionary=None, hit_names=None):
    """
    plot convolutions and their ranks (optional).

    Parameters
    ----------
    model : keras model
    ofname : string
        output filename prefix
    rank_dictionary : dictionary of ranks, optional
        rank_dictionary['auROC'][j] is the rank of convolution
        j based on contribution to auROC.
    hit_names : list of string lists
        hit_names[i][j] is pwn name for jth match to ith convolution
    """
    weights, _ = model.layers[0].get_weights()
    num_conv, _, _, conv_width = weights.shape
    if rank_dictionary is not None:
        assert all(len(ranks)==num_conv for ranks in rank_dictionary.values()), \
        "convolution ranks dont match number of convolutions"
    weights.squeeze()
    if hit_names is not None:
        text_title = '>Top encode pwm matches:\n\n'
        text_pos = np.shape(weights)[-1] + 1
    absolute_weights = np.absolute(weights)
    for i, filter_weights in enumerate(absolute_weights):
        plt.clf()
        plot_bases(filter_weights.T)
        if hit_names is not None:
            text_string = text_title+'\n'.join(hit_names[i])
            plt.text(text_pos, 0, text_string)
        if rank_dictionary is not None:
            rank_strings = [' : '.join(
                [metric, str(rank_dictionary[metric][i])])
                    for metric in rank_dictionary.keys()]
            plt.title("Convolution %s \n%s" %
                      (str(i), '\n'.join(rank_strings)) )
        else:
            plt.title("Convolution %s" % str(i))
        figure = plt.gcf()
        figure.savefig("%s.convolution_%s.png" % (ofname, str(i)),
                       bbox_inches='tight')
    return

def get_sequence_dl_scores(model, X, batch_size=200, num_tasks=1):
    """
    Assumes single task Sequential keras model with final activation layer.

    Parameters
    ----------
    model: keras model
    X: ndarray

    Returns
    ------
    input_scores: ndarray
    """
    from keras.layers.core import Activation
    assert isinstance(model.layers[-1], Activation), \
    "scoring requires final activation layer!"
    from deeplift import keras_conversion as kc

    kc.mean_normalise_first_conv_layer_weights(model)
    deeplift_model = kc.convert_sequential_model(model)
    target_contribs_func = deeplift_model.get_target_contribs_func(input_layer_idx=0)
    return np.asarray(target_contribs_func(task_idx=2, input_data_list=[X],
                                            batch_size=batch_size, progress_update=10000))
