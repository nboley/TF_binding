import os
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
 
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
