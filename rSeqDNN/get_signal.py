import numpy as np
from pybedtools import Interval
import wWigIO

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
