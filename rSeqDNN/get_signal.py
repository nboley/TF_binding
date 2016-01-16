import numpy as np
from pybedtools import Interval
import wWigIO
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
    data : 3d array
        shaped (N, 4, L) where L is number of sequence
        and L is sequence length.
    """
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    data = 0.25 * np.ones((len(peaks), pk_width, 4), dtype='float32')
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
    data.swapaxes(1,2)

    return data

def _bigwig_extractor(datafile, intervals, **kwargs):
    width = intervals[0].stop - intervals[0].start
    data = np.zeros((len(intervals), 1, 1, width))

    wWigIO.open(datafile)

    for index, interval in enumerate(intervals):
        wWigIO.getData(datafile, interval.chrom, interval.start, interval.stop,
                       data[index, 0, 0, :])

    wWigIO.close(datafile)

    return data

class BigwigExtractor(object):
    multiprocessing_safe = True

    def __init__(self, datafile, **kwargs):
        self._datafile = datafile
        self._halfwidth = kwargs.get('local_norm_halfwidth', None)

    def __call__(self, intervals, **kwargs):
        if self._halfwidth:
            width = intervals[0].stop - intervals[0].start
            offset = width // 2 - self._halfwidth

            slopped_intervals = [
                Interval(interval.chrom,
                         interval.start + offset,
                         interval.stop - offset)
                for interval in intervals
            ]

            data = _bigwig_extractor(self._datafile, slopped_intervals,
                                     **kwargs)
            mean = data.mean(axis=3, keepdims=True)
            std = data.std(axis=3, keepdims=True)

            return (data[:, 0, 0, -offset:-offset + width] - mean) / std
        else:
            return _bigwig_extractor(self._datafile, intervals, **kwargs)

def encode_peaks_bigwig_into_array(peaks, bigwig_fname, batch_size=1000):
    """
    Extracts sequence input arrays.

    Parameters
    ----------
    peaks : sequence of NarrowPeak
    bigwig_fname : str
    batch_size : int, default: 1000
        Determines batching during bigwig loading.
    """
    bw = BigwigExtractor(bigwig_fname)
    data_batches = [bw(get_intervals_from_peaks(peaks_batch))
                    for peaks_batch in batch_iter(peaks, batch_size)]
    data = np.concatenate(data_batches)

    return data
