import os
import gzip
from collections import namedtuple
from itertools import izip
import random
import cPickle as pickle

import numpy as np
from sklearn.cross_validation import StratifiedKFold

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
    'NarrowPeak', ['contig', 'start', 'stop', 'summit', 
                   'score', 'signalValue', 'pValue', 'qValue', 'idrValue', 'seq'])
NarrowPeakData.__new__.__defaults__ = (None,) * len(NarrowPeakData._fields)

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

PeakAndLabel = namedtuple('PeakAndLabel', ['peak', 'sample', 'label', 'score'])
class PeaksAndLabels():
    def __getitem__(self, index):
        return PeakAndLabel(
            self.peaks[index], 
            self.samples[index], 
            self.labels[index], 
            self.scores[index])
    
    def __iter__(self):
        return (
            PeakAndLabel(pk, sample, label, score) 
            for pk, sample, label, score
            in izip(self.peaks, self.samples, self.labels, self.scores)
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
        self.scores = []
        self.sample_ids = set()
        self.contigs = set()
        self.peak_widths = set()
        for pk, sample, label, score in peaks_and_labels:
            self.peaks.append(pk)
            self.peak_widths.add(pk.pk_width)
            self.samples.append(sample)
            self.labels.append(label)
            self.scores.append(score)
            self.sample_ids.add(sample)
            self.contigs.add(pk.contig)
        assert len(self.peak_widths) == 1
        # turn the list of labels into a numpy array
        self.labels = np.array(self.labels, dtype='float32')
        self.labels.flags.writeable = False
        self.scores = np.array(self.scores, dtype='float32')
        self.scores.flags.writeable = False
        
    def subset_data(self, sample_names, contigs):
        '''return data covering sample+names and contigs
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.sample in sample_names
                and pk_and_label.peak.contig in contigs
            )

    def remove_data(self, sample_names, contigs):
        '''return data not covering sample_names and contigs                                                                                                                             
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self
                if pk_and_label.sample not in sample_names
                and pk_and_label.peak.contig not in contigs
            )

    def remove_ambiguous_labeled_entries(self):
        '''return subset of data wityh nonzero labels
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.label != -1
            )

    def iter_train_validation_subsets(
            self, validation_contigs=None, single_celltype=False):
        for train_indices, valid_indices in iter_train_validation_splits(
                self.sample_ids, self.contigs,
                validation_contigs, single_celltype):
            yield (self.subset_data(*train_indices),
                   self.subset_data(*valid_indices))

class FastaPeaksAndLabels(PeaksAndLabels):
    @staticmethod
    def __name__():
        return 'FastaPeaksAndLabels'

    def subset_data(self, subset_indices):
        return FastaPeaksAndLabels(
            self[index] for index in subset_indices)

    def iter_train_validation_subsets(self):
        skf = StratifiedKFold(self.labels, n_folds=5)
        for train_indices, valid_indices in skf:
            yield (self.subset_data(train_indices),
                   self.subset_data(valid_indices))

def iter_summit_centered_peaks(original_peaks, half_peak_width):
    for peak in original_peaks:
        centered_peak = NarrowPeak(
            peak.contig, 
            peak.start+peak.summit-half_peak_width, 
            peak.start+peak.summit+half_peak_width,
            half_peak_width, 
            peak.score,
            peak.signalValue,
            peak.pValue,
            peak.qValue,
            peak.idrValue)
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

        try: summit = int(data[9])
        except IndexError: summit = (stop-start)/2

        try: score = float(data[4])
        except IndexError: score = -1.0
        try: signalValue = float(data[6])
        except IndexError: signalValue = -1.0
        try: pValue = float(data[7])
        except IndexError: pValue = -1.0
        try: qValue = float(data[8])
        except IndexError: qValue = -1.0
        # idr Value's dont exist in narrowPeakFiles
        idrValue = -1.0
        seq = None
        
        yield NarrowPeak(
            chrm, start, stop, summit, 
            score, signalValue, pValue, qValue, idrValue, seq)

    return

def load_labeled_peaks_from_beds(
        pos_regions_fp, neg_regions_fp, 
        half_peak_width=None):
    def iter_all_pks():        
        for pos_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(pos_regions_fp), half_peak_width):
            yield PeakAndLabel(pos_pk, 'sample', 1, pos_pk.signalValue)
        for neg_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(neg_regions_fp), half_peak_width):
            yield PeakAndLabel(neg_pk, 'sample', 0, neg_pk.signalValue)
    return PeaksAndLabels(iter_all_pks())

def iter_fasta(fp, max_n_peaks=None):
    '''
    convert fasta data into NarrowPeak
    '''
    if isinstance(fp, str):
        raise ValueError, "Expecting filepointer"
    
    all_lines = [line for line in fp]
    score = -1.0
    signalValue = -1.0
    pValue = -1.0
    qValue = -1.0
    idrValue = -1.0
    start = 0
    def parse_seq_list(seq_list, start=0):
        seq = ''.join(seq_list)
        stop = len(seq)
        return seq, stop, int((stop-start)/2)
    name, seq_list = None, []
    for i, line in enumerate(all_lines):
        if max_n_peaks != None and i > max_n_peaks:
            break
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                seq, stop, summit = parse_seq_list(seq_list)
                yield NarrowPeak(
                    name, start, stop, summit,
                    score, signalValue, pValue, qValue, idrValue,
                    seq)
            name, seq_list = line, []
        else:
            seq_list.append(line)
    if name is not None and (max_n_peaks == None or i > max_n_peaks):
            seq, stop, summit = parse_seq_list(seq_list)
            yield NarrowPeak(
                name, start, stop, summit,
                score, signalValue, pValue, qValue, idrValue, seq)

def load_labeled_peaks_from_fastas(
        pos_sequences_fp, neg_sequences_fp,
        max_num_peaks_per_sample=None):
    def iter_all_seqs():        
        for pos_pk in iter_fasta(pos_sequences_fp, 
                                 max_num_peaks_per_sample):
            assert pos_pk.pk_width==len(pos_pk.seq)
            yield PeakAndLabel(pos_pk, 'sample', 1, pos_pk.signalValue)
        for neg_pk in iter_fasta(neg_sequences_fp,
                                 max_num_peaks_per_sample):
            assert neg_pk.pk_width==len(neg_pk.seq)
            yield PeakAndLabel(neg_pk, 'sample', 0, neg_pk.signalValue)
    peaks_and_labels = FastaPeaksAndLabels(iter_all_seqs())
    return peaks_and_labels

chipseq_peaks_tabix_file_cache = {}
def label_and_score_peak_with_chipseq_peaks(
        chipseq_peak_fnames, 
        peak, 
        min_overlap_frac=0.5,
        score_index=6 # defaults to signal value
    ):
    """Label peaks by which ChIPseq peaks they overlap.

    Score is set to the maximum score over all overlapping peaks, or 0 if 
    no peaks overlap. 
    """
    pid = os.getppid()
    peak_coords = (peak.contig, 
                   peak.start, 
                   peak.stop)
    labels = []
    scores = []
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
            labels.append(-1)
            scores.append(0)
            continue
        overlap_frac = 0.0
        score = 0
        for chipseq_pk in fp.fetch(*peak_coords):
            data = chipseq_pk.split()
            c_start, c_stop = data[1:3]
            overlap = (
                min(peak.stop, int(c_stop)) - max(peak.start, int(c_start)))
            pk_overlap_frac = float(overlap)/(int(c_stop) - int(c_start))
            if pk_overlap_frac > overlap_frac:
                overlap_frac = pk_overlap_frac
                score = data[score_index]
        
        if overlap_frac > min_overlap_frac:
            labels.append(1)
            scores.append(score) # XXX
        else:
            labels.append(0)
            scores.append(0)
    return labels, scores

def iter_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        tf_id, 
        annotation_id,
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        include_ambiguous_peaks=False):
    # put the import here to avoid errors if the database isn't available
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        tf_id, annotation_id)
    for (sample_index, (sample_id, (
            sample_chipseq_peaks_fnames, dnase_peaks_fnames)
                )) in enumerate(peak_fnames.iteritems()):
        # for now, dont allow multiple set of DNASE peaks
        assert len(dnase_peaks_fnames) == 1
        dnase_peaks_fname = next(iter(dnase_peaks_fnames))
        
        optimal_sample_chipseq_peaks_fnames = sample_chipseq_peaks_fnames[
            'optimal idr thresholded peaks']
        ambiguous_sample_chipseq_peak_fnames = sample_chipseq_peaks_fnames[
            'anshul relaxed peaks']
        # if we're not using ambiuous peaks and there are no optimal peaks,
        # then skip this samples
        if ( not include_ambiguous_peaks 
             and len(optimal_sample_chipseq_peaks_fnames) == 0):
            continue
        # if there aren't any peak files then skip this samples
        if ( len(ambiguous_sample_chipseq_peak_fnames) == 0 
             and len(optimal_sample_chipseq_peaks_fnames) == 0):
            continue

        # try to use anshul's relaxed peaks for the relaxed peak set.
        if ( include_ambiguous_peaks 
             and len(ambiguous_sample_chipseq_peak_fnames) == 0):
            continue
        
        print "Loading peaks for sample '%s' (%i/%i)" % (
            sample_id, sample_index, len(peak_fnames))
        with getFileHandle(dnase_peaks_fname) as fp:
            pks_iter = iter_narrow_peaks(fp)
            if half_peak_width != None:
                pks_iter = iter_summit_centered_peaks(pks_iter, half_peak_width)
            num_peaks = 0
            for i, pk in enumerate(pks_iter):
                labels, scores = label_and_score_peak_with_chipseq_peaks(
                    optimal_sample_chipseq_peaks_fnames, pk)
                # set the label to zero if there is no clean peak set
                # (in this case all peaks will be labeled -1 or 1 )
                # aggregate labels by taking the max over all labels
                label, score = 0, 0
                if len(labels) > 0:
                    assert len(scores) > 0
                    label = max(labels)
                    score = max(scores)
                if include_ambiguous_peaks and label == 0:
                    (relaxed_labels, relaxed_scores
                         ) = label_and_score_peak_with_chipseq_peaks(
                             ambiguous_sample_chipseq_peak_fnames, pk)
                    if max(relaxed_labels) == 1:
                        label = -1
                        score = max(relaxed_scores)
                
                yield PeakAndLabel(pk, sample_id, label, score)
                num_peaks += 1
                if ( max_n_peaks_per_sample is not None 
                     and num_peaks >= max_n_peaks_per_sample): 
                    break
    return

def load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        tf_id, 
        annotation_id,
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        include_ambiguous_peaks=False):
    """
    
    """
    # check for a pickled file int he current directory
    pickle_fname = "peaks_and_label.%s.%s.%s.%s.obj" % (
        tf_id, half_peak_width, 
        max_n_peaks_per_sample, 
        include_ambiguous_peaks)
    try:
        with open(pickle_fname) as fp:
            print "Using pickled peaks_and_labels from '%s'." % pickle_fname 
            return pickle.load(fp)
    except IOError:
        pass
    peaks_and_labels = PeaksAndLabels(
        iter_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            tf_id, 
            annotation_id,
            half_peak_width, 
            max_n_peaks_per_sample, 
            include_ambiguous_peaks))
    with open(pickle_fname, "w") as ofp:
        pickle.dump(peaks_and_labels, ofp)
    return peaks_and_labels
