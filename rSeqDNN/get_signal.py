import numpy as np
from pybedtools import Interval
import bigWigFeaturize
from pyTFbindtools.sequence import one_hot_encode_sequence
from pyTFbindtools.peaks import get_intervals_from_peaks

def batch_iter(iterable, batch_size):
    '''iterates in batches.
    '''
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in xrange(batch_size):
                values += (it.next(),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values

def encode_peaks_sequence_into_array(peaks, fasta):
    """
    Extracts sequence input arrays.

    Parameters
    ----------
    peaks : sequence of NarrowPeak
    fasta : FastaFile

    Returns
    -------
    data : 4d array
        shaped (N, 1, 4, L) where N is number of sequences
        and L is sequence length.
    """
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    data = 0.25 * np.ones((len(peaks), 1, pk_width, 4), dtype='float32')
    for i, pk in enumerate(peaks):
        if pk.seq is not None:
            seq = pk.seq
        else:
            seq = fasta.fetch(pk.contig, pk.start, pk.stop)
        # skip sequences overrunning the contig boundary
        if len(seq) != pk_width: continue
        coded_seq = one_hot_encode_sequence(seq)
        data[i] = coded_seq
    # swap base and position axes
    data = data.swapaxes(2,3)

    return data

def encode_peaks_bigwig_into_array(peaks, bigwig_fnames, cache=None):
    """
    Extracts bigwig input arrays.

    Parameters
    ----------
    peaks : sequence of NarrowPeak
    bigwig_fnames : list
        Expects list of bigwig filenames.

    Returns
    -------
    4d array shaped (N, 1, k, L) where N is number of regions,
    k is the number of bigwig filenames, and L is sequence length.
    """
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    if isinstance(cache, basestring):
        return bigWigFeaturize.new(bigwig_fnames, pk_width,
                                   intervals=get_intervals_from_peaks(peaks), cache=cache)
    else:
        return bigWigFeaturize.new(bigwig_fnames, pk_width,
                                   intervals=get_intervals_from_peaks(peaks))

def get_peaks_signal_arrays(peaks, genome_fasta, bigwig_fname,
                            reverse_complement=False):
    """
    Get sequence of signal arrays.
    """
    signal_arrays = []
    if genome_fasta is not None:
        print('loading features from fasta...')
        sequence_array = encode_peaks_sequence_into_array(peaks, genome_fasta)
        if reverse_complement:
            sequence_array = np.concatenate((sequence_array,
                                             sequence_array[:, :, ::-1, ::-1]))
        signal_arrays.append(sequence_array)
    if bigwig_fname is not None:
        print('loading features from bigwig...')
        bigwig_array = encode_peaks_bigwig_into_array(peaks, [bigwig_fname])
        if reverse_complement:
            bigwig_array = np.concatenate((bigwig_array, bigwig_array[:, :, :, ::-1]))
        signal_arrays.append(bigwig_array)

    return signal_arrays
