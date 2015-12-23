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

ENCODE_MOTIFS_PATH =  os.path.join(os.getcwd(),'encode_motifs.txt.gz')

def score_convolutions(model, X, batch_size):
    """
    Score convolutions using deepLifft on maxpooling layer.
    Limited to architectures consisting of convolution+maxpool+sigmoid.

    Parameters
    ----------
    model : keras model
        Restricted to models consisting of convolution+maxpool+sigmoid 
    X : 4d array
        one hot encoded sequences.

    Returns
    -------
    convolution_scores : 3d array
        (N, K, L) array with scores on N sequences of K convolutions
        with L maxpool windows.
    """
    scripts_dir = os.environ.get("ENHANCER_SCRIPTS_DIR")
    sys.path.insert(0, "%s/featureSelector/deepLIFFT" % scripts_dir)
    from modelSaver import CNNMaxPoolIPModel
    import layers
    conv_weights, conv_biases = model.layers[0].get_weights()
    maxpool_size = model.layers[1].pool_size[1]
    ip_weights = model.layers[-1].get_weights()[0].T
    ip_biases = model.layers[-1].get_weights()[1]
    prop = CNNMaxPoolIPModel(conv_weights, conv_biases,
                             True, None, maxpool_size, True,
                             ip_weights, ip_biases,
                             layers.OutputActivations.sigmoid)
    blobName = "pool1"
    prop.setFirstInputBlobsToTrackNames([blobName])
    deepLifft_score_list = []
    for start in xrange(0, len(X), batch_size):
        if (start+batch_size)>len(X):
                end  = len(X)
        else:
                end = start + batch_size
        prop.updateInputLayers(np.expand_dims(X[start:end, :, :, :], axis=0))
        prop.performFprop()
        prop.performBackprop()
        outputBlobName = prop.globalBlobsLookup.outputBlobToTrackName
        outputBlob = prop.globalBlobsLookup.getBlob(outputBlobName)
        blob = prop.globalBlobsLookup.getBlob(blobName)
        deepLifft_scores, _, _ = layers.getVariousDeepLifftScoresFromBlob_batch(
                blob, outputBlob, prop.inputBlobNameToOutputActivation[outputBlobName])
        deepLifft_score_list.append(deepLifft_scores.squeeze(axis=(1,3)))
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

def plot_convolutions(model, ofname, rank_dictionary=None):
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
    """
    scripts_dir = os.environ.get("UTIL_SCRIPTS_DIR")
    weights, biases = model.layers[0].get_weights()
    num_conv, _, _, conv_width = weights.shape
    if rank_dictionary is not None:
        assert all(len(ranks)==num_conv for ranks in rank_dictionary.values()), \
        "convolution ranks dont match number of convolutions"
    reshaped_weights = np.reshape(weights, (num_conv, 4, conv_width))
    temp_fname = "tempFile.txt"
    for i in xrange(num_conv):
        plt.clf()
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        np.savetxt(temp_fname, reshaped_weights[i].T, delimiter='\t')
        png_fname = "%s.%s.png" % (ofname, str(i))
        os.system("Rscript %s/logoViz/plotConvFilter.R %s %s %s" % (
            scripts_dir, temp_fname, png_fname, str(biases[i])))
        plt.imshow(mpimg.imread(png_fname))
        if rank_dictionary is not None:
            rank_strings = [' : '.join(
                [metric, str(rank_dictionary[metric][i])])
                    for metric in rank_dictionary.keys()]
            plt.title("Convolution %s \n%s" %
                      (str(i), '\n'.join(rank_strings)) )
        else:
            plt.title("Convolution %s" % str(i))
        plt.savefig("%s.convolution_%s.png" % (ofname, str(i)))
        os.system("rm %s" % png_fname)
    os.system("rm %s" % temp_fname)
    return
