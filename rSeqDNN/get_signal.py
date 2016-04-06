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

def encode_peaks_bigwig_into_array(peaks, bigwig_fnames, cache=None,
                                   local_norm_halfwidth=500):
    """
    Extracts bigwig input arrays.

    Parameters
    ----------
    peaks : sequence of NarrowPeak
    bigwig_fnames : list
        Expects list of bigwig filenames.
    cache : path, default: cwd
        Bigwig features cached here.
    local_norm_halfwidth : int, default: 5000
        Window halfwidth context used to normalize signal with z-scores.
        Note: must be greater then peak width.

    Returns
    -------
    4d array shaped (N, 1, k, L) where N is number of regions,
    k is the number of bigwig filenames, and L is sequence length.
    """
    # find the peak width
    pk_width = peaks[0].pk_width
    # make sure that the peaks are all the same width
    assert all(pk.pk_width == pk_width for pk in peaks)
    print('num of peaks: %i' % len(peaks))
    intervals=get_intervals_from_peaks(peaks)
    print('num of intervals: %i' % len(intervals))
    if local_norm_halfwidth:
            width = intervals[0].stop - intervals[0].start
            offset = width // 2 - local_norm_halfwidth
            slopped_intervals = [Interval(interval.chrom,
                                          interval.start + offset,
                                          interval.stop - offset)
                                 for interval in intervals]
            if isinstance(cache, basestring):
                data = bigWigFeaturize.new(bigwig_fnames, 2*local_norm_halfwidth,
                                           intervals=slopped_intervals, cache=cache)
            else:
                data = bigWigFeaturize.new(bigwig_fnames, 2*local_norm_halfwidth,
                                           intervals=slopped_intervals)
            assert len(data)==len(peaks), "Num of signal regions: %i!" % len(data)
            mean = data.mean(axis=3, keepdims=True)
            std = data.std(axis=3, keepdims=True)
            data_norm = (data[:, :, :, -offset:(-offset+width)] - mean) / std
            return data_norm
    else:
        if isinstance(cache, basestring):
            return bigWigFeaturize.new(bigwig_fnames, pk_width,
                                       intervals=intervals, cache=cache)
        else:
            return bigWigFeaturize.new(bigwig_fnames, pk_width,
                                       intervals=intervals)

def get_peaks_signal_arrays(peaks, genome_fasta, bigwig_fnames,
                            reverse_complement=False):
    """
    Get sequence of signal arrays.

    Parameters
    ----------
    peaks: sequence of NarrowPeak
    genome_fasta: FastaFile
    bigwig_fnames: sequence
    reverse_complement: boolean
    """
    signal_arrays = []
    if genome_fasta is not None:
        print('loading features from fasta...')
        sequence_array = encode_peaks_sequence_into_array(peaks, genome_fasta)
        if reverse_complement:
            sequence_array = np.concatenate((sequence_array,
                                             sequence_array[:, :, ::-1, ::-1]))
        signal_arrays.append(sequence_array)
    for bigwig_fname in bigwig_fnames:
        print('loading features from bigwig %s' % bigwig_fname)
        bigwig_array = encode_peaks_bigwig_into_array(peaks, [bigwig_fname])
        if reverse_complement:
            bigwig_array = np.concatenate((bigwig_array, bigwig_array[:, :, :, ::-1]))
        signal_arrays.append(bigwig_array)

    return signal_arrays

def get_peaks_signal_arrays_by_samples(peaks_and_labels, genome_fasta, sample_dependent_bigwigs,
                                       sample_independent_bigwigs=None,
                                       reverse_complement=False,
                                       return_labels=False, return_scores=False):
    """
    Get sequence of signal arrays for each sample in PeaksAndLabels.

    Parameters
    ----------
    peaks_and_labels : PeaksAndLabels obj
    genome_fasta : FastaFile
    sample_dependent_bigwigs: dict
        dictionary with sample names as keys and lists of
        sample-specific filenames as values.
    sample_independent_bigwigs: sequence, optional
        sample independent bigwig signals (e.g. conservation).
    reverse_complement : boolean, default: false
    return_labels : boolean, default: false
    return_scores : boolean, default: false

    Returns
    -------
    results, a list with sequence of signal arrays
    and, optionally, labels and scores.
    """
    ## check number of bigwigs per sample is constant
    num_bigwigs = len(sample_dependent_bigwigs.values()[0])
    assert all(len(bigwigs)==num_bigwigs for bigwigs in sample_dependent_bigwigs.values())
    ## TODO: logic is convoluted, needs refactoring
    results = []
    per_sample_signal_arrays = []
    per_sample_labels = []
    per_sample_scores = []
    samples = peaks_and_labels.sample_ids
    contigs = peaks_and_labels.contigs
    for sample in samples:
        peaks_and_labels_sample = peaks_and_labels.subset_data([sample], contigs)
        print 'loading sample %s' % sample
        bigwig_fnames = sample_dependent_bigwigs[sample]
        if sample_independent_bigwigs is not None:
            for bigwig_fname in sample_independent_bigwigs:
                bigwig_fnames.append(bigwig_fname)
        per_sample_signal_arrays.append(get_peaks_signal_arrays(peaks_and_labels_sample.peaks,
                                                                genome_fasta, bigwig_fnames,
                                                                reverse_complement=reverse_complement))
        if return_labels:
            if reverse_complement:
                per_sample_labels.append(np.concatenate((peaks_and_labels_sample.labels,
                                                         peaks_and_labels_sample.labels)))
            else:
                per_sample_labels.append(peaks_and_labels_sample.labels)
        if return_scores:
            if reverse_complement:
                per_sample_scores.append(np.concatenate((peaks_and_labels_sample.scores,
                                                         peaks_and_labels_sample.scores)))
            else:
                per_sample_scores.append(peaks_and_labels_sample.scores)
    if len(per_sample_signal_arrays)>1:
        per_sample_signal_arrays = np.asarray(per_sample_signal_arrays)
        signal_arrays = [np.concatenate(per_sample_signal_arrays[:, i])
                         for i in xrange(len(per_sample_signal_arrays[0, :]))]
        results.append(signal_arrays)
        if return_labels:
            results.append(np.concatenate(per_sample_labels))
        if return_scores:
            results.append(np.concatenate(per_sample_scores))
    else:
        signal_arrays = per_sample_signal_arrays[0]
        results.append(signal_arrays)
        if return_labels:
            results.append(per_sample_labels[0])
        if return_scores:
            results.append(per_sample_scores[0])

    return results

def merge_sample_specific_bigwigs(bigwig_dictionaries):
    """
    Merge bigwig dictionaries.

    Parameters
    ----------
    bigwig_dictionaries: sequence of dict
        each dict contains sample names as keys and
        lists of filenames as values.

    Returns
    -------
    sample_specific_bigwigs
    """
    pass
