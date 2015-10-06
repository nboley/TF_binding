import os
import gzip
from collections import namedtuple
from itertools import izip
import random

import numpy as np

from pysam import TabixFile

from grit.lib.multiprocessing_utils import Counter

from cross_validation import iter_train_validation_splits

def getFileHandle(filename, mode="r"):
    if filename.endswith('.gz') or filename.endswith('.gzip'):
        if (mode=="r"):
            mode="rb";
        return gzip.open(filename,mode)
    else:
        return open(filename,mode)

NarrowPeakData = namedtuple(
    'NarrowPeak', ['contig', 'start', 'stop', 'summit', 'score'])

class NarrowPeak(NarrowPeakData):
    @property
    def identifier(self):
        return "{0.contig}:{0.start}-{0.stop}_{0.summit}".format(self)
    @property    
    def pk_width(self):
        return self.stop - self.start

class Peaks(list):
    pass

class PeaksAndLabelsThreadSafeIterator(object):
    def __init__(self, peaks_and_labels):
        self.peaks_and_labels = peaks_and_labels
        self.i = Counter()
        self.n = len(peaks_and_labels)
        self._cur_val = 0

    def __iter__(self):
        return self

    def next(self):
        assert self.n == len(self.peaks_and_labels)
        i = self.i.return_and_increment()
        self._cur_val = i
        if i < self.n:
            return self.peaks_and_labels[i]
        else:
            raise StopIteration()

PeakAndLabel = namedtuple('PeakAndLabel', ['peak', 'sample', 'label'])
class PeaksAndLabels():
    def __getitem__(self, index):
        return PeakAndLabel(
            self.peaks[index], self.samples[index], self.labels[index])
    
    def __iter__(self):
        return (
            PeakAndLabel(pk, sample, label) 
            for pk, sample, label 
            in izip(self.peaks, self.samples, self.labels)
        )

    def thread_safe_iter(self):
        """Returns an iterator that can safely be used from multiple threads.

        """
        return PeaksAndLabelsThreadSafeIterator(self)

    def __len__(self):
        rv = len(self.peaks)
        assert len(self.samples) == rv
        assert len(self.labels) == rv
        return rv
    
    @property
    def max_peak_width(self):
        return max(self.peak_widths)
    
    def __init__(self, peaks_and_labels):
        # split the peaks and labels into separate columns. Also
        # keep track of the distinct samples and contigs
        self.peaks = []
        self.samples = []
        self.labels = []
        self.sample_ids = set()
        self.contigs = set()
        self.peak_widths = set()
        for pk, sample, label in peaks_and_labels:
            self.peaks.append(pk)
            self.peak_widths.add(pk.pk_width)
            self.samples.append(sample)
            self.labels.append(label)
            self.sample_ids.add(sample)
            self.contigs.add(pk.contig)
        assert len(self.peak_widths) == 1
        # turn the list of labels into a numpy array
        self.labels = np.array(self.labels, dtype=int)
        
    def subset_data(self, sample_names, contigs):
        '''return data covering sample+names and contigs
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.sample in sample_names
                and pk_and_label.peak.contig in contigs
            )

    def iter_train_validation_subsets(self):
        for train_indices, valid_indices in iter_train_validation_splits(
                self.sample_ids, self.contigs):
            yield (self.subset_data(*train_indices),
                   self.subset_data(*valid_indices))

def iter_summit_centered_peaks(
        original_peaks, half_peak_width, 
        max_n_peaks=None, retain_score=False):
    for peak in original_peaks:
        centered_peak = NarrowPeak(
            peak.contig, 
            peak.start+peak.summit-half_peak_width, 
            peak.start+peak.summit+half_peak_width,
            half_peak_width, 
            peak.score if retain_score else -1)
        # skip peaks that are too close to the contig start
        if centered_peak.start <= 0: continue
        yield centered_peak

    return

def iter_narrow_peaks(fp, max_n_peaks=None):
    if isinstance(fp, str):
        raise ValueError, "Expecting filepointer"

    all_lines = [line for line in fp]
    random.shuffle(all_lines)
    for i, line in enumerate(all_lines):
        if line.startswith("track"): continue
        if max_n_peaks != None and i > max_n_peaks: 
            break
        data = line.split()
        chrm = data[0]
        start = int(data[1])
        stop = int(data[2])
        try: score = float(data[6])
        except IndexError: score = -1
        try: summit = int(data[9])
        except IndexError: summit = (stop-start)/2
        yield NarrowPeak(chrm, start, stop, summit, score)

    return

def load_labeled_peaks_from_beds(
        pos_regions_fp, neg_regions_fp, half_peak_width=None):
    def iter_all_pks():        
        for pos_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(pos_regions_fp), half_peak_width):
            yield PeakAndLabel(pos_pk, 'sample', 1)
        for neg_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(neg_regions_fp), half_peak_width):
            yield PeakAndLabel(neg_pk, 'sample', 0)
    return PeaksAndLabels(iter_all_pks())


chipseq_peaks_tabix_file_cache = {}
def classify_chipseq_peak(chipseq_peak_fnames, peak, min_overlap_frac=0.5):
    pid = os.getppid()
    peak_coords = (peak.contig, 
                   peak.start, 
                   peak.stop)
    status = []
    for fname in chipseq_peak_fnames:
        # get the tabix file pointer, opening it if necessary 
        try: 
            fp = chipseq_peaks_tabix_file_cache[(pid, fname)]
        except KeyError:
            fp = TabixFile(fname)
            chipseq_peaks_tabix_file_cache[(pid, fname)] = fp

        # if the contig isn't in the contig list, then it
        # can't be a valid peak
        if peak.contig not in fp.contigs: 
            status.append(0)
            continue
        overlap_frac = 0.0
        for chipseq_pk in fp.fetch(*peak_coords):
            c_start, c_stop = chipseq_pk.split()[1:3]
            overlap = (
                min(peak.stop, int(c_stop)) - max(peak.start, int(c_start)))
            overlap_frac = max(
                overlap_frac, 
                float(overlap)/(int(c_stop) - int(c_start))
            )
        
        if overlap_frac > min_overlap_frac:
            status.append(1)
            continue
        else:
            status.append(0)
    return status

def iter_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        tf_id, 
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        skip_ambiguous_peaks=False):
    # put the import here to avoid errors if the database isn't available
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        tf_id)    
    for (sample_index, (sample_id, (sample_chipseq_peaks_fnames, dnase_peaks_fnames)
            )) in enumerate(peak_fnames.iteritems()):
        # for now, dont allow multiple set of DNASE peaks
        assert len(dnase_peaks_fnames) == 1
        dnase_peaks_fname = next(iter(dnase_peaks_fnames))

        optimal_sample_chipseq_peaks_fnames = sample_chipseq_peaks_fnames[
            'optimal idr thresholded peaks']
        noisy_sample_chipseq_peaks_fnames = []
        for peaks_type, fnames in sample_chipseq_peaks_fnames.iteritems():
            if peaks_type == 'optimal idr thresholded peaks': continue
            noisy_sample_chipseq_peaks_fnames.extend(fnames)
                
        print "Loading peaks for sample '%s' (%i/%i)" % (
            sample_id, sample_index, len(peak_fnames))
        with getFileHandle(dnase_peaks_fname) as fp:
            pks_iter = iter_narrow_peaks(fp)
            if half_peak_width != None:
                pks_iter = iter_summit_centered_peaks(pks_iter, half_peak_width)
            num_peaks = 0
            for i, pk in enumerate(pks_iter):
                label = classify_chipseq_peak(
                    optimal_sample_chipseq_peaks_fnames, pk)
                # merge labels
                label = max(label)
                if skip_ambiguous_peaks and label == 0:
                    noisy_label = classify_chipseq_peak(
                        noisy_sample_chipseq_peaks_fnames, pk)
                    if max(noisy_label) == 1:
                        print "NOISY", pk
                        continue
                yield PeakAndLabel(pk, sample_id, label)
                num_peaks += 1
                if ( max_n_peaks_per_sample is not None 
                     and num_peaks >= max_n_peaks_per_sample): 
                    break
    return

def load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        tf_id, 
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        skip_ambiguous_peaks=False):
    """
    
    """
    return PeaksAndLabels(
        iter_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            tf_id, 
            half_peak_width, 
            max_n_peaks_per_sample, 
            skip_ambiguous_peaks))
