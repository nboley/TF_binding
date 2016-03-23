import os
import math
import gzip
from collections import namedtuple, defaultdict, OrderedDict
from itertools import izip
import random
import cPickle as pickle
import re
import hashlib

import numpy as np


from sklearn.cross_validation import StratifiedKFold

from pysam import Tabixfile, FastaFile

from pyDNAbinding.sequence import one_hot_encode_sequence

from grit.lib.multiprocessing_utils import Counter
from grit.files.reads import ChIPSeqReads, MergedReads

from cross_validation import iter_train_validation_splits

from pyDNAbinding.misc import optional_gzip_open, load_fastq
from pyDNAbinding.binding_model import FixedLengthDNASequences

from pyTFbindtools.DB import load_tf_ids, load_tf_names_for_sample

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

def upsample(seqs, num_seqs):
    new_seqs = []
    new_seqs.extend(seqs)
    while len(new_seqs) < num_seqs:
        new_seqs.extend(seqs[:num_seqs-len(new_seqs)])
    return new_seqs

def encode_peaks_sequence_into_binary_array(peaks, fasta):
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
        data[i,:,:] = coded_seq
    return data

class NarrowPeak(NarrowPeakData):
    @property
    def identifier(self):
        return "{0.contig}:{0.start}-{0.stop}_{0.summit}".format(self)
    @property    
    def pk_width(self):
        return self.stop - self.start

class Peaks(list):
    pass

class ThreadSafeIterator(object):
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

class TrainValidationthreadSafeIterator(object):
    def __init__(self, peaks_and_labels):
        self.peaks_and_labels = peaks_and_labels
        tr_valid_indices = list(iter_train_validation_splits(
                self.sample_ids, self.contigs))
        self.i = Counter()
        self.n = len(tr_valid_indices)
        self._cur_val = 0

    def __iter__(self):
        return self

    def next(self):
        i = self.i.return_and_increment()
        self._cur_val = i
        if i < self.n:
            train_indices, valid_indices = self.tr_valid_indices[i]
            return (self.peaks_and_labels.subset_data(*train_indices),
                    self.peaks_and_labels.subset_data(*valid_indices))
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
                and (contigs is None or pk_and_label.peak.contig in contigs)
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

    def thread_safe_iter_train_validation_subsets(self):
        return TrainValidationthreadSafeIterator(self)

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

    for i, line in enumerate(fp):
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
            fp = Tabixfile(fname)
            chipseq_peaks_tabix_file_cache[(pid, fname)] = fp

        # if the contig isn't in the contig list, then it
        # can't be a valid peak
        if peak.contig not in fp.contigs: 
            labels.append(0)
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

def iter_sample_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        annotation_id,
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        include_ambiguous_peaks=False,
        order_by_accessibility=False):
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, tf_id, )

    pass

def build_peaks_label_mat(
        annotation_id, roadmap_sample_id, half_peak_width=None):
    # load all of the peaks
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, roadmap_sample_id=roadmap_sample_id)[roadmap_sample_id]
    assert not any(x is None for x in peak_fnames.keys())
    # extract the accessibility peaks
    dnase_peaks_fnames = peak_fnames['dnase']
    print annotation_id, roadmap_sample_id
    assert len(dnase_peaks_fnames) == 1
    dnase_peaks_fname = next(iter(dnase_peaks_fnames))
    del peak_fnames['dnase']

    # load the DNASE peaks
    print "Loading peaks"
    with getFileHandle(dnase_peaks_fname) as fp:
        if half_peak_width is None:
            pks = list(iter_narrow_peaks(fp))
        else:
            pks = list(iter_summit_centered_peaks(
                iter_narrow_peaks(fp), half_peak_width))
    all_labels = []
    for desired_peak_type in (
            'optimal idr thresholded peaks', 'anshul relaxed peaks'):
        # load the labels - using a cached version when available
        pickle_fname = "PICKLEDLABELSMAT_%i_%s_%s_%s.%s.obj" % (
            annotation_id, 
            roadmap_sample_id,
            ('IDROPTIMAL' if desired_peak_type=='optimal idr thresholded peaks'
             else 'RELAXEDPEAKS'),
            half_peak_width,
            hashlib.sha1(str(sorted(list(peak_fnames.keys())))).hexdigest(),
        )
        try:
            with open(pickle_fname) as fp:
                print "Loading labels"
                labels = np.load(fp)
        except IOError:
            print "Building labels - this could take a while"
            labels = np.zeros((len(pks), len(peak_fnames)), dtype='float32')
            for tf_index, (tf_id, chipseq_peaks_fnames) in enumerate(
                    sorted(peak_fnames.iteritems())):
                optimal_chipseq_peaks_fnames = [
                    fname for pk_type, fname in chipseq_peaks_fnames
                    if pk_type == desired_peak_type
                ]
                for i, pk in enumerate(pks):
                    pk_labels, pk_scores = label_and_score_peak_with_chipseq_peaks(
                        optimal_chipseq_peaks_fnames, pk)
                    try: label = max(pk_labels)
                    except ValueError: label = 0 # if there are no overlapping peaks
                    labels[i, tf_index] = label
                    if i>0 and i%100000 == 0:
                        print "%i/%i peaks, %i/%i samples" % (
                            i, len(pks), tf_index, len(peak_fnames))
            with open(pickle_fname + ".tfnames", "w") as ofp:
                for tf_name in sorted(list(peak_fnames.keys())):
                    ofp.write("%s\n" % tf_name)
            with open(pickle_fname, "w") as ofp:
                np.save(ofp, labels)
        all_labels.append(labels)

    pk_record_type = type(pks[0])
    pk_types = ('S64', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S')
    pks = np.array(pks, dtype=zip(pks[0]._fields, pk_types))

    return pks, sorted(peak_fnames.keys()), all_labels

def iter_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
        tf_id, 
        annotation_id,
        half_peak_width=None, 
        max_n_peaks_per_sample=None,
        include_ambiguous_peaks=False,
        order_by_accessibility=False):
    # put the import here to avoid errors if the database isn't available
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    all_peaks = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, tf_id)
    for (sample_index, (sample_id, sample_peak_fnames)
            ) in enumerate(all_peaks.iteritems()):
        sample_chipseq_peaks_fnames = sample_peak_fnames[tf_id]
        dnase_peaks_fnames = sample_peak_fnames['dnase']
        # for now, dont allow multiple set of DNASE peaks
        assert len(dnase_peaks_fnames) == 1
        dnase_peaks_fname = next(iter(dnase_peaks_fnames))
        
        optimal_sample_chipseq_peak_fnames = [
            fname for pk_type, fname in sample_chipseq_peaks_fnames
            if pk_type == 'optimal idr thresholded peaks'
        ]
        ambiguous_sample_chipseq_peak_fnames = [
            fname for pk_type, fname in sample_chipseq_peaks_fnames
            if pk_type == 'anshul relaxed peaks'
        ]
        # if we're not using ambiuous peaks and there are no optimal peaks,
        # then skip this samples
        if ( not include_ambiguous_peaks 
             and len(optimal_sample_chipseq_peak_fnames) == 0):
            continue
        # if there aren't any peak files then skip this samples
        if ( len(ambiguous_sample_chipseq_peak_fnames) == 0 
             and len(optimal_sample_chipseq_peak_fnames) == 0):
            continue
        # try to use anshul's relaxed peaks for the relaxed peak set.
        if ( include_ambiguous_peaks 
             and len(ambiguous_sample_chipseq_peak_fnames) == 0):
            continue
        
        print "Loading peaks for sample '%s' (%i/%i)" % (
            sample_id, sample_index, len(all_peaks))
        with getFileHandle(dnase_peaks_fname) as fp:
            pks_iter = list(iter_narrow_peaks(fp))
            if order_by_accessibility:
                pks_iter.sort(key=lambda x:-x.signalValue)
            if half_peak_width != None:
                pks_iter = iter_summit_centered_peaks(pks_iter, half_peak_width)
            num_peaks = 0
            for i, pk in enumerate(pks_iter):
                labels, scores = label_and_score_peak_with_chipseq_peaks(
                    optimal_sample_chipseq_peak_fnames, pk)
                assert all(label in (0,1) for label in labels)
                # set the label to -1 if there is no clean peak set
                # (in this case all peaks will be labeled 0 or 1 )
                # aggregate labels by taking the max over all labels
                label, score = 0, 0
                # set the label to the max label over the clean peaks
                if len(labels) > 0:
                    assert len(scores) > 0
                    label = max(labels)
                    score = max(scores)
                # if there is not an overlapping clean peak, then see if there 
                # is an overlapping ambiguous peak. If so, then label the region
                # as ambiguous (no clean peak, but a dirty peak)
                if include_ambiguous_peaks and label <= 0:
                    (relaxed_labels, relaxed_scores
                         ) = label_and_score_peak_with_chipseq_peaks(
                             ambiguous_sample_chipseq_peak_fnames, pk)
                    assert all(label in (0,1) for label in relaxed_labels)
                    # if there is a relaxed peak with label 1, then this 
                    # contradicts the clean peaks so we label it as ambiguous 
                    if max(relaxed_labels) == 1:
                        label = -1
                        score = max(score, max(relaxed_scores))
                    # otherwise this is not labelled, so we assume that this
                    # is not covered and thus remains a 0 (although now clean
                    # because it's not overlapped by a noisy peak)
                assert include_ambiguous_peaks or label != -1
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
        include_ambiguous_peaks=False,
        order_by_accessibility=False):
    """
    
    """
    # check for a pickled file int he current directory
    pickle_fname = "peaks_and_label.%s.%s.%s.%s.%s.obj" % (
        tf_id, half_peak_width, 
        max_n_peaks_per_sample, 
        include_ambiguous_peaks,
        order_by_accessibility)
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
            include_ambiguous_peaks,
            order_by_accessibility=order_by_accessibility))
    with open(pickle_fname, "w") as ofp:
        pickle.dump(peaks_and_labels, ofp)
    return peaks_and_labels


def load_accessibility_data(sample_ids, pks):
    from pyTFbindtools.DB import load_dnase_fnames
    # get the correct filename
    fnames = load_dnase_fnames(sample_ids)
    
    pk_width = pks[0]['stop'] - pks[0]['start'] + 1
    
    cached_fname = "cachedaccessibility.%s.%s.obj" % (
        hashlib.sha1(self.pks.view(np.uint8)).hexdigest(),
        hash(tuple(fnames))
    )
    try:
        with open(cached_fname) as fp:
            print "Loading cached accessibility data"
            rv = np.load(fp)
    except IOError:
        rv = bigWigFeaturize.new(
            fnames,
            pk_width, 
            intervals=[
                Region(pk['contig'], pk['start'], pk['stop']) for pk in self.pks
            ]
        )

        with open(cached_fname, "w") as ofp:
            print "Saving accessibility data"
            np.save(ofp, rv)

    return rv

def load_chipseq_coverage(sample_id, tf_id, peaks):
    from pyTFbindtools.DB import load_chipseq_fnames
    cached_fname = "cachedtfcov.%s.%s.%s.obj" % (
        hashlib.sha1(peaks.view(np.uint8)).hexdigest(),
        sample_id, 
        tf_id
    )
    try:
        with open(cached_fname) as fp:
            print "Loading cached chipseq coverage for %s, %s (%s)" % (
                sample_id, tf_id, cached_fname)
            rv = np.load(fp)
    except IOError:
        rv = np.zeros(
            (2, len(peaks), peaks[0][2] - peaks[0][1]), 
            dtype='float32')
        return rv
        fnames, control_fnames = load_chipseq_fnames(sample_id, tf_id)
        fnames = [x.replace("/mnt/lab_data/kundaje/users/nboley/TF_binding/", 
                            "/srv/scratch/nboley/cached_data/")
                  for x in fnames]
        control_fnames = [
            x.replace("/mnt/lab_data/kundaje/users/nboley/TF_binding/", 
                      "/srv/scratch/nboley/cached_data/")
            for x in control_fnames
        ]
        for fname in fnames+control_fnames:
            try: open(fname)
            except IOError: print fname
        #assert False
        reads = MergedReads([
            ChIPSeqReads(fname).init() for fname in fnames])
        control_reads = MergedReads([
            ChIPSeqReads(fname).init() for fname in control_fnames])
        for i, pk in enumerate(peaks):
            cov = reads.build_read_coverage_array(
                pk[0], '.', pk[1], pk[2]-1)
            bg_cov = control_reads.build_read_coverage_array(
                pk[0], '.', pk[1], pk[2]-1)
            rv[0,i,:] = cov
            rv[1,i,:] = bg_cov
            if i%1000 == 0: print i, len(peaks), cov.shape
        with open(cached_fname, "w") as ofp:
            print "Saving cached chipseq coverage for %s, %s" % (
                sample_id, tf_id)
            np.save(ofp, rv)

    return rv

def load_DNASE_coverage(sample_id, peaks):
    from pyTFbindtools.DB import load_dnase_fnames
    cached_fname = "cacheddnasecov.%s.%s.obj" % (
        hashlib.sha1(peaks.view(np.uint8)).hexdigest(),
        sample_id
    )
    try:
        with open(cached_fname) as fp:
            print "Loading cached DNASE coverage for %s (%s)" % (
                sample_id, cached_fname)
            rv = np.load(fp)
    except IOError:
        rv = np.zeros(
            (1, len(peaks), peaks[0][2] - peaks[0][1]), 
            dtype='float32')
        fnames = load_dnase_fnames(sample_id)
        reads = MergedReads([
            ChIPSeqReads(fname).init() for fname in fnames])
        for i, pk in enumerate(peaks):
            cov = reads.build_read_coverage_array(
                pk[0], '.', pk[1], pk[2]-1)
            rv[0,i,:] = cov
            if i%1000 == 0: print i, len(peaks), cov.shape
        with open(cached_fname, "w") as ofp:
            print "Saving cached DNASE coverage for %s (%s)" % (
                sample_id, cached_fname)
            np.save(ofp, rv)
    return rv


class SamplePeaksAndLabels():
    @staticmethod
    def one_hot_code_peaks_sequence(pks, genome_fasta):
        cached_fname = "cachedseqs.%s.obj" % hashlib.sha1(
            pks.view(np.uint8)).hexdigest()
        try:
            with open(cached_fname) as fp:
                print "Loading cached seqs", cached_fname
                return np.load(fp)
        except IOError:
            pass

        pk_width = pks[0][2] - pks[0][1]
        rv = 0.25 * np.ones((len(pks), pk_width, 4), dtype='float32')
        for i, data in enumerate(pks):
            assert pk_width == data[2] - data[1]
            seq = genome_fasta.fetch(str(data[0]), data[1], data[2])
            if len(seq) != pk_width: continue
            coded_seq = one_hot_encode_sequence(seq)
            rv[i,:,:] = coded_seq
        
        # add the extra dimension for theano
        rv = np.swapaxes(rv, 1, 2)[:,None,:,:]
        
        with open(cached_fname, "w") as ofp:
            print "Saving seqs"
            np.save(ofp, rv)

        return rv

    @property
    def factor_names(self):
        from pyTFbindtools.DB import load_tf_names
        if self._factor_names is None:
            self._factor_names = [
                x.replace("eGFP-", "") for x in load_tf_names(self.tf_ids)]
            assert len(self.tf_ids) == len(self._factor_names)
        return self._factor_names

    def subset_pks(self, pk_indices=slice(None), factor_indices=slice(None)):
        """Return a copy of self only containing the peaks sepcified by indices

        indices: numpy array of indices to include
        """
        
        #assert False
        return SamplePeaksAndLabels(
            self.sample_id,
            np.array(self.tf_ids)[factor_indices],
            self.pks[pk_indices], 
            self.fwd_seqs[pk_indices],
            self.idr_optimal_labels[pk_indices, factor_indices], 
            self.relaxed_labels[pk_indices, factor_indices]
        )

    def balance_data(self):
        labels = self.labels
        one_indices = np.random.choice(
            np.nonzero(labels == 1)[0], size=len(labels)/2)
        zero_indices = np.random.choice(
            np.nonzero(labels == 0)[0], size=len(labels)/2)
        indices = np.concatenate((one_indices, zero_indices), axis=0)
        np.random.shuffle(indices)
        return self.subset_pks(pk_indices=indices)
    
    def subset_tfs(self, desired_factor_names):
        """Return a copy of self that only contains factor_names.

        """
        # make chained filtering more convenient by defaulting to 
        # all tfs if none is passed
        if desired_factor_names is None:
            return self

        # keep the top n_samples peaks and max_num_tfs tfs
        all_factor_names = self.factor_names
        # make sure all of the tfnames actually exist
        #assert ( desired_factor_names is None 
        #         or all(factor_name in self.factor_names 
        #                for factor_name in desired_factor_names) 
        #)
        # filter the tf set 
        filtered_tfs = sorted([
            (factor_name, tf_id, i) for i, (factor_name, tf_id) 
            in enumerate(zip(all_factor_names, self.tf_ids)) 
            if factor_name in desired_factor_names
        ])
        factor_names, tf_ids, tf_indices = zip(*filtered_tfs)

        return self.subset_pks(factor_indices=np.array(tf_indices))

    def subset_pks_by_rank(self, max_num_peaks, use_top_accessible, seed=0):
        """Return a copy of self containing at most max_num_peaks peaks.

        max_num_peaks: the maximum number of peaks to return
        use_top_accessible: return max_num_peaks most accessible peaks
        """
        if max_num_peaks is None:
            max_num_peaks = len(self.pks)
        
        # sort the peaks by accessibility
        if use_top_accessible:
            indices = np.lexsort((self.pks['start'], -self.pks['signalValue']))
        # sort the peaks randomly
        else:
            # set a seed so we can use cached peaks between debug rounds
            np.random.seed(seed)
            indices = np.argsort(np.random.random(len(self.pks)))
        return self.subset_pks(pk_indices=indices[:max_num_peaks])
    
    def subset_pks_by_contig(
            self, contigs_to_include=None, contigs_to_exclude=None):
        assert (contigs_to_include is None) != (contigs_to_exclude is None), \
            "Either contigs_to_include or contigs_to_exclude must be specified"
        # case these to sets to speed up the search a little
        if contigs_to_include is not None:
            contigs_to_include = set(contigs_to_include)
        if contigs_to_exclude is not None:
            contigs_to_exclude = set(contigs_to_exclude)
        indices = np.array([
            i for i, pk in enumerate(self.pks) 
            if (contigs_to_exclude is None or pk[0] not in contigs_to_exclude)
            and (contigs_to_include is None or pk[0] in contigs_to_include)
        ])
        return self.subset_pks(pk_indices=indices)

    def save(self, cached_fname_suffix):
        print "Saving cached peaks and labels"
        np.save("pks." + cached_fname_suffix, self.pks)
        np.save("fwd_seqs." + cached_fname_suffix, self.fwd_seqs)
        np.save("idr_optimal_labels." + cached_fname_suffix, 
                self.idr_optimal_labels)
        np.save("relaxed_labels." + cached_fname_suffix, self.relaxed_labels)

    @staticmethod
    def load(cached_fname_suffix, sample_id, tf_ids):
        print "Loading cached peaks and labels"
        with open("pks.%s.npy"%cached_fname_suffix) as fp: 
            pks = np.load(fp)
        with open("fwd_seqs.%s.npy"%cached_fname_suffix) as fp: 
            fwd_seqs = np.load(fp)
        with open("idr_optimal_labels.%s.npy"%cached_fname_suffix) as fp: 
            idr_optimal_labels = np.load(fp)
        with open("relaxed_labels.%s.npy"%cached_fname_suffix) as fp:
            relaxed_labels = np.load(fp)
        print "FINISHED Loading cached peaks and labels"
        return SamplePeaksAndLabels(
            sample_id, 
            tf_ids, 
            pks, 
            fwd_seqs, 
            idr_optimal_labels, 
            relaxed_labels
        )

    @property
    def chipseq_coverage(self):
        if self._chipseq_coverage is None:
            self._chipseq_coverage = np.concatenate([
                load_chipseq_coverage(self.sample_id, tf_id, self.pks)
                for tf_id in self.tf_ids
            ], axis=1)
            # normalize the coverage
            sums = self._chipseq_coverage.sum(axis=2).sum(axis=1)
            self._chipseq_coverage[0] = 1e6*self._chipseq_coverage[0]/sums[0]
            self._chipseq_coverage[1] = 1e6*self._chipseq_coverage[1]/sums[1]
        return self._chipseq_coverage

    @property
    def dnase_coverage(self):
        if self._dnase_coverage is None:
            self._dnase_coverage = np.concatenate([
                load_DNASE_coverage(self.sample_id, self.pks)
            ], axis=1)
            print "DNASE SHAPE:", self._dnase_coverage.shape
            # normalize the coverage
            sums = self._dnase_coverage.sum(axis=1).sum(axis=1)
            self._dnase_coverage[0] = 1e6*self._dnase_coverage[0]/sums[0]
        return self._dnase_coverage

    @property
    def n_samples(self):
        return int(self.clean_labels.shape[0])

    def __init__(self, 
                 sample_id, tf_ids, 
                 pks, seqs, 
                 idr_optimal_labels, relaxed_labels):
        self.sample_id = sample_id
        self.tf_ids = tf_ids
        self._factor_names = None

        self.pks = pks
        self.fwd_seqs = seqs
        self.seq_length = self.fwd_seqs.shape[3]

        self._chipseq_coverage = None
        self._dnase_coverage = None
        
        self.idr_optimal_labels = idr_optimal_labels
        self.relaxed_labels = relaxed_labels
        
        # set the ambiguous labels to -1
        self.ambiguous_pks_mask = (
            self.idr_optimal_labels != self.relaxed_labels)
        self.clean_labels = self.idr_optimal_labels.copy()
        self.clean_labels[self.ambiguous_pks_mask] = -1

        self.labels = self.clean_labels # idr_optimal_labels # ambiguous_labels
        assert self.labels.shape[1] == len(self.tf_ids)

    def build_balanced_indices(self):
        one_indices = np.random.choice(
            np.nonzero(self.labels == 1)[0], size=len(self.labels)/2)
        zero_indices = np.random.choice(
            np.nonzero(self.labels == 0)[0], size=len(self.labels)/2)
        permutation = np.concatenate((one_indices, zero_indices), axis=0)
        np.random.shuffle(permutation)
        return permutation

    def build_shuffled_indices(self):
        return np.random.permutation(self.labels.shape[0])

    def build_ordered_indices(self):
        return np.arange(self.labels.shape[0])

    def iter_batches_from_indices(self, 
                                  batch_size, 
                                  repeat_forever, 
                                  indices_generator, 
                                  include_chipseq_signal=False, 
                                  include_dnase_signal=False):
        i = 0
        n = int(math.ceil(self.fwd_seqs.shape[0]/float(batch_size)))
        permutation = None
        while repeat_forever is True or i<n:
            if i%n == 0:
                permutation = indices_generator()
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            indices = permutation[subset]
            rv =  {'fwd_seqs': self.fwd_seqs[indices], 
                   'output': self.labels[indices]
            }
            if include_dnase_signal:
                rv['dnase_cov'] = self.dnase_coverage[0,indices][:,None,None,:]
            if include_chipseq_signal:
                rv['chipseq_cov'] = (
                    np.swapaxes(self.chipseq_coverage[:,indices], 0, 1))[:,:,None,:]
            yield rv
            i += 1
        return
    
    def iter_batches(self, 
                     batch_size, 
                     repeat_forever, 
                     balanced=False, 
                     shuffled=False, 
                     **kwargs):
        if balanced:
            indices_generator = self.build_balanced_indices
        elif shuffled:
            indices_generator = self.build_shuffled_indices
        else:
            indices_generator = self.build_ordered_indices
        
        return self.iter_batches_from_indices(
            batch_size, 
            repeat_forever, 
            indices_generator,
            **kwargs
        )

class PartitionedSamplePeaksAndLabels():
    def cache_key(self, sample_id):
        return hashlib.sha1(str((
            sample_id, 
            tuple(sorted(self.factor_names)), 
            self.n_samples, 
            self.annotation_id, 
            self.half_peak_width
        ))).hexdigest()

    def _save_cached(self, sample_id):
        self.data[sample_id].save(self.cache_key(sample_id) + ".data.obj")
        self.train[sample_id].save(self.cache_key(sample_id) + ".train.obj")
        self.validation[sample_id].save(self.cache_key(sample_id) + ".validation.obj")
        return
    
    def _load_cached(self, sample_id):
        self.tf_ids = load_tf_ids(self.factor_names)
        self.data[sample_id] = None
        #SamplePeaksAndLabels.load(
        #    self.cache_key + ".data.obj", sample_id, tf_ids)
        self.train[sample_id] = SamplePeaksAndLabels.load(
            self.cache_key(sample_id)+".train.obj", sample_id, self.tf_ids)
        self.validation[sample_id] = SamplePeaksAndLabels.load(
            self.cache_key(sample_id)+".validation.obj", sample_id, self.tf_ids)
        assert (self.train[sample_id].factor_names 
                == self.validation[sample_id].factor_names)
        assert (self.train[sample_id].tf_ids 
                == self.validation[sample_id].tf_ids)

    @staticmethod
    def _load_data(roadmap_sample_id, 
                   factor_names, 
                   n_samples, 
                   annotation_id, 
                   half_peak_width):
        # XXX search for cached data
        pks, tf_ids, (idr_optimal_labels, relaxed_labels) = build_peaks_label_mat(
            annotation_id, roadmap_sample_id, half_peak_width)
        print "Coding peaks"
        from pyDNAbinding.DB import load_genome_metadata
        genome_fasta = FastaFile('hg19.genome.fa')
        # load_genome_metadata(annotation_id).filename)
        fwd_seqs = SamplePeaksAndLabels.one_hot_code_peaks_sequence(
            pks, genome_fasta)
        
        print "Filtering Peaks"
        data = SamplePeaksAndLabels(
            roadmap_sample_id, tf_ids, 
            pks, fwd_seqs, 
            idr_optimal_labels, relaxed_labels
        )
        data = data.subset_pks_by_rank(
            max_num_peaks=n_samples, use_top_accessible=False
        )
        data = data.subset_tfs(factor_names)
        return data

    def __init__(self, 
                 roadmap_sample_ids, 
                 factor_names, 
                 n_samples=None, 
                 annotation_id=1, 
                 half_peak_width=500):
        self.sample_ids = roadmap_sample_ids
        if factor_names is None:
            factor_names = set()
            for x in map(load_tf_names_for_sample, roadmap_sample_ids):
                factor_names.update(x)
        self.factor_names = sorted(factor_names)
        self.n_samples = n_samples
        self.annotation_id = annotation_id
        self.half_peak_width = half_peak_width
        self.seq_length = 2*half_peak_width

        self.data = {}
        self.train = {}
        self.validation = {}
        for sample_id in self.sample_ids:
            try: 
                #raise IOError, "TEST"
                self._load_cached(sample_id)
            except IOError:
                self.data[sample_id] = self._load_data(
                    sample_id, 
                    self.factor_names, 
                    self.n_samples, 
                    self.annotation_id, 
                    self.half_peak_width)
                assert self.data[sample_id].seq_length == self.seq_length
                print "Splitting out train data"        
                self.train[sample_id] = self.data[
                    sample_id].subset_pks_by_contig(
                        contigs_to_exclude=('chr1', 'chr2', 'chr8', 'chr9')
                    )
                print "Splitting out validation data"        
                self.validation[sample_id] = self.data[
                    sample_id].subset_pks_by_contig(
                        contigs_to_include=('chr8', 'chr9')
                    )
                self._save_cached(sample_id)

            assert (self.train[sample_id].factor_names 
                    == self.validation[sample_id].factor_names)
            #self.factor_names = self.train.factor_names

    def iter_batches(self, batch_size, data_subset, repeat_forever, **kwargs):
        ## determine the batch sizes
        if data_subset == 'train':
            data_subset = self.train
        elif data_subset == 'validation':
            data_subset = self.validation
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        ## find the number of observations to sample from each batch
        # To make this work, I would need to randomly choose the extra observations
        assert batch_size >= len(data_subset), "Cant have a batch size smaller than the number of samples"
        fractions = np.array([x.n_samples for x in data_subset.values()], dtype=float)
        fractions = fractions/fractions.sum()
        inner_batch_sizes = np.array(batch_size*fractions, dtype=int)
        # accounting for any rounding from the previous step 
        for i in xrange(batch_size - inner_batch_sizes.sum()):
            inner_batch_sizes[i] += 1

        ### XXXXX - PUT THIS IN THE ACTUAL GENERATOR
        ## build the sample labels array
        sample_labels = np.zeros(
            (batch_size, len(data_subset)), dtype='float32')
        for i in xrange(len(data_subset)):
            start_index = (0 if i == 0 else np.cumsum(inner_batch_sizes)[i-1])
            stop_index = np.cumsum(inner_batch_sizes)[i]
            sample_labels[start_index:stop_index,i] = 1
        iterators = OrderedDict(
            (sample_id, 
             data.iter_batches(i_batch_size, repeat_forever, **kwargs) )
            for i_batch_size, (sample_id, data) in zip(
                    inner_batch_sizes, data_subset.iteritems())
        )

        def f():
            while True:
                grpd_res = defaultdict(list)
                cnts = []
                for sample_id, iterator in iterators.iteritems():
                    data = next(iterator)
                    cnt = None
                    for key, vals in data.iteritems():
                        grpd_res[key].append(vals)
                        if cnt == None: cnt = vals.shape[0]
                        assert cnt == vals.shape[0]
                    cnts.append(cnt)
                for key, vals in grpd_res.iteritems():
                    grpd_res[key] = np.concatenate(grpd_res[key], axis=0)
                    
                # build the sample labels
                cnts = np.array(cnts)
                sample_labels = np.zeros(
                    (cnts.sum(), len(iterators)), dtype='float32')
                for i in xrange(len(data_subset)):
                    start_index = (0 if i == 0 else np.cumsum(cnts)[i-1])
                    stop_index = np.cumsum(inner_batch_sizes)[i]
                    sample_labels[start_index:stop_index,i] = 1                
                assert 'sample_ids' not in grpd_res
                grpd_res['sample_ids'] = sample_labels
                
                # cast thios to a normal dict (rather than a default dict)
                yield dict(grpd_res)
            return
        
        return f()

    def iter_train_data(
            self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(batch_size, 'train', repeat_forever, **kwargs)

    def iter_validation_data(
            self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(
            batch_size, 'validation', repeat_forever, **kwargs)

class SelexDBConn(object):
    def __init__(self, host, dbname, user, exp_id):
        import psycopg2
        self.conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % (host, dbname, user))
        self.exp_id = exp_id
        return

    def get_factor_name(self):
        cur = self.conn.cursor()
        query = "SELECT tf_name FROM selex_experiments NATURAL JOIN tfs WHERE selex_exp_id = '%s';"
        cur.execute(query, (self.exp_id,))
        res = cur.fetchall()
        return res[0][0]

    def get_factor_id(self):
        cur = self.conn.cursor()
        query = "SELECT tf_id FROM selex_experiments WHERE selex_exp_id = '%s';"
        cur.execute(query, (self.exp_id,))
        res = cur.fetchall()
        return res[0][0]

    def insert_model_into_db(
            self, mo, validation_lhd):
        assert isinstance(mo, EnergeticDNABindingModel)
        motif_len = mo.motif_len
        cur = self.conn.cursor()
        query = """
        INSERT INTO new_selex_models
          (selex_exp_id, motif_len, consensus_energy, ddg_array, validation_lhd)
          VALUES 
          (%s, %s, %s, %s, %s) 
        RETURNING key
        """
        cur.execute(query, (
            self.exp_id, 
            motif_len, 
            float(mo.ref_energy), 
            mo.ddg_array.tolist(),
            float(validation_lhd)
        ))
        self.conn.commit()
        return

    def get_primer_and_fnames(self):
        cur = self.conn.cursor()
        query = """
        SELECT rnd, primer, fname 
          FROM selex_round
         WHERE selex_exp_id = '%i'
         ORDER BY rnd;
        """
        cur.execute(query % self.exp_id)
        primers = set()
        fnames = {}
        for rnd, primer, fname in cur.fetchall():
            fnames[int(rnd)] = fname
            primers.add(primer)

        assert len(primers) == 1
        primer = list(primers)[0]
        query = """
        SELECT fname 
          FROM selex_background_sequences
         WHERE primer = '%s';
        """
        cur.execute(query % primer)
        res = cur.fetchall()
        if len(res) == 0:
            bg_fname = None
        else:
            bg_fname = res[0][0]
        
        return primer, fnames, bg_fname

    def get_fnames(self):
        return self.get_primer_and_fnames()[1:]

    def get_dna_and_prot_conc(self):
        cur = self.conn.cursor()
        query = """
        SELECT dna_conc, prot_conc 
          FROM selex_round
         WHERE selex_exp_id = '%i';
        """
        cur.execute(query % self.exp_id)
        concs = set(cur.fetchall())
        assert len(concs) == 1
        return concs.pop()

def load_sequences(fnames, max_num_seqs_per_file=float('inf')):
    fnames = list(fnames)
    rnds_and_seqs = {}
    rnd_nums = [int(x.split("_")[-1].split(".")[0]) for x in fnames]
    rnds_and_fnames = dict(zip(rnd_nums, fnames))
    for rnd, fname in rnds_and_fnames.iteritems():
        with optional_gzip_open(fname) as fp:
            loader = load_fastq if ".fastq" in fname else load_text_file
            rnds_and_seqs[rnd] = loader(fp, max_num_seqs_per_file)
    return rnds_and_seqs

def load_sequence_data(selex_db_conn,
                       seq_fps,
                       background_seq_fp, 
                       max_num_seqs_per_file, 
                       min_num_background_sequences):
    if selex_db_conn is not None:
        assert seq_fps is None and background_seq_fp is None
        fnames, bg_fname = selex_db_conn.get_fnames()
        seq_fps = [ optional_gzip_open(fname) for fname in fnames.values() ]
        background_seq_fp = optional_gzip_open(bg_fname)

    pyTFbindtools.log("Loading sequences", 'VERBOSE')
    rnds_and_seqs = load_sequences(
        (x.name for x in seq_fps), max_num_seqs_per_file)

    """
    if  max_num_seqs_per_file < min_num_background_sequences:
        pyTFbindtools.log(
            "WARNING: reducing the number of background sequences to --max-num-seqs-per-file ")
        min_num_background_sequences = max_num_seqs_per_file
    else:
        min_num_background_sequences = min_num_background_sequences
    """
    min_num_background_sequences = max(
        min_num_background_sequences, 
        max(len(seqs) for seqs in rnds_and_seqs.values())
    )
    background_seqs = None
    if background_seq_fp is not None:
        with optional_gzip_open(background_seq_fp.name) as fp:
            background_seqs = load_fastq(fp, max_num_seqs_per_file)
    else:
        background_seqs = sample_random_seqs(
            min_num_background_sequences, 
            len(rnds_and_seqs.values()[0][0]))

    if len(background_seqs) < min_num_background_sequences:
        pyTFbindtools.log(
            "Too few (%i) background sequences were provided - sampling an additional %i uniform random sequences" % (
                len(background_seqs), 
                min_num_background_sequences-len(background_seqs)
            ), "VERBOSE")
        seq_len = len(rnds_and_seqs.values()[0][0])
        assert len(background_seqs) == 0 or len(background_seqs[0]) == seq_len,\
            "Background sequence length does not match sequence length."
        background_seqs.extend( 
            sample_random_seqs(
                min_num_background_sequences-len(background_seqs), seq_len)
        )

    return rnds_and_seqs, background_seqs

class SelexExperiment():
    @property
    def id(self):
        selex_exp_id = self.selex_exp_id
        factor_name = self.factor_name
        factor_id = self.factor_id
        return 'invitro_%s_%s_%s' % (
            factor_name, factor_id, selex_exp_id)

    @property
    def seq_length(self):
        return self.fwd_seqs.shape[3]

    def get_coded_seqs_and_labels(self, unbnd_seqs, bnd_seqs, primer):
        left_flank, right_flank = re.split('[0-9]{1,4}N', primer)
        left_flank = left_flank.rjust(self.seq_pad).replace(' ', 'N')
        right_flank = right_flank.ljust(self.seq_pad).replace(' ', 'N')

        num_seqs = min(len(unbnd_seqs), len(bnd_seqs))
        unbnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in unbnd_seqs[:num_seqs]], 
            include_shape=False)
        bnd_seqs = FixedLengthDNASequences(
            [left_flank + x + right_flank for x in bnd_seqs[:num_seqs]], 
            include_shape=False)

        permutation = np.random.permutation(2*num_seqs)
        fwd_one_hot_seqs = np.vstack(
            (unbnd_seqs.fwd_one_hot_coded_seqs, bnd_seqs.fwd_one_hot_coded_seqs)
        )[permutation,:,:]
        fwd_one_hot_seqs = np.swapaxes(fwd_one_hot_seqs, 1, 2)[:,None,:,:]

        if unbnd_seqs.fwd_shape_features is None:
            shape_seqs = None
        else:
            fwd_shape_seqs = np.vstack(
                (unbnd_seqs.fwd_shape_features, bnd_seqs.fwd_shape_features)
            )[permutation,:,:]
            rc_shape_seqs = np.vstack(
                (unbnd_seqs.rc_shape_features, bnd_seqs.rc_shape_features)
            )[permutation,:,:]
            shape_seqs = np.concatenate(
                (fwd_shape_seqs, rc_shape_seqs), axis=2)[:,None,:,:]

        labels = np.hstack(
            (np.zeros(len(unbnd_seqs), dtype='float32'), 
             np.ones(len(bnd_seqs), dtype='float32'))
        )[permutation]
        
        return fwd_one_hot_seqs, shape_seqs, labels # rc_one_hot_seqs, 

    def __init__(
            self, selex_exp_id, n_samples, validation_split=0.1, seq_pad=40):
        self.selex_exp_id = selex_exp_id
        self.n_samples = n_samples
        self.seq_pad = seq_pad
        self.validation_split = validation_split

        # load connect to the DB, and find the factor name
        selex_db_conn = SelexDBConn(
            'mitra', 'cisbp', 'nboley', selex_exp_id)
        self.factor_name = selex_db_conn.get_factor_name()
        self.factor_id = selex_db_conn.get_factor_id()

        # load the sequencess
        print "Loading sequences for %s-%s (exp ID %i)" % (
            self.factor_name, self.factor_id, self.selex_exp_id)
        self.primer, fnames, bg_fname = selex_db_conn.get_primer_and_fnames()
        if bg_fname is None:
            raise NoBackgroundSequences(
                "No background sequences for %s (%i)." % (
                    self.factor_name, self.selex_exp_id))

        rnd_num = max(fnames.keys())
        with optional_gzip_open(fnames[rnd_num]) as fp:
            bnd_seqs = load_fastq(fp, self.n_samples)
        with optional_gzip_open(bg_fname) as fp:
            unbnd_seqs = load_fastq(fp, self.n_samples)
        if len(bnd_seqs[0]) < 20:
            raise SeqsTooShort("Seqs too short for %s (exp %i)." % (
                self.factor_name, selex_exp_id))
        
        if self.n_samples is not None:
            if len(unbnd_seqs) < self.n_samples:
                unbnd_seqs = upsample(unbnd_seqs, self.n_samples)
            if len(bnd_seqs) < self.n_samples:
                bnd_seqs = upsample(bnd_seqs, self.n_samples)

        (self.fwd_seqs, self.shape_seqs, self.labels 
             ) = self.get_coded_seqs_and_labels(
                 unbnd_seqs, bnd_seqs, self.primer)
        self.n_samples = self.fwd_seqs.shape[0]
        self._training_index = int(self.n_samples*self.validation_split)
        self.train_fwd_seqs = self.fwd_seqs[self._training_index:]
        self.train_labels = self.labels[self._training_index:]
        
        self.validation_fwd_seqs = self.fwd_seqs[:self._training_index]
        self.validation_labels = self.labels[:self._training_index]
        
        return

    def iter_batches(
            self, batch_size, data_subset, repeat_forever, **kwargs):
        if data_subset == 'train': 
            fwd_seqs = self.train_fwd_seqs
            labels = self.train_labels
        elif data_subset == 'validation':
            fwd_seqs = self.validation_fwd_seqs
            labels = self.validation_labels
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        i = 0
        n = fwd_seqs.shape[0]//batch_size
        if n <= 0: raise ValueError, "Maximum batch size is %i (requested %i)" \
           % (fwd_seqs.shape[0], batch_size)
        while repeat_forever is True or i < n:
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            yield {'fwd_seqs': fwd_seqs[subset], 'output': labels[subset,None]}
            i += 1
        
        return
    
    def iter_train_data(self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(batch_size, 'train', repeat_forever)

    def iter_validation_data(self, batch_size, repeat_forever=False, **kwargs):
        return self.iter_batches(batch_size, 'validation', repeat_forever)

class SelexData():
    def load_tfs_grpd_by_family(self):
        raise NotImplementedError, "This fucntion just exists to save teh sql query"
        query = """
          SELECT tfs.family_id, family_name, array_agg(tfs.factor_name) 
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN tf_families 
        GROUP BY family_id, family_name;
        """ # group tfs by their families

        pass

    @staticmethod
    def find_all_selex_experiments():
        import psycopg2
        conn = psycopg2.connect(
            "host=%s dbname=%s user=%s" % ('mitra', 'cisbp', 'nboley'))
        cur = conn.cursor()    
        query = """
          SELECT selex_experiments.tf_id, tfs.tf_name, array_agg(distinct selex_exp_id)
            FROM selex_experiments NATURAL JOIN tfs NATURAL JOIN roadmap_matched_chipseq_peaks
        GROUP BY selex_experiments.tf_id, tfs.tf_name
        ORDER BY tf_name;
        """
        cur.execute(query)
        # filter out the TFs that dont have background sequence or arent long enough
        res = [ x for x in cur.fetchall() 
                if x[1] not in ('E2F1', 'E2F4', 'POU2F2', 'PRDM1', 'RXRA')
                #and x[1] in ('MAX',) # 'YY1', 'RFX5', 'USF', 'PU1', 'CTCF')
        ]
        return res

    def add_selex_experiment(self, selex_exp_id):
        self.experiments.append(
            SelexExperiment(
                selex_exp_id, self.n_samples, self.validation_split)
        )

    def add_all_selex_experiments_for_factor(self, factor_name):
        exps_added = 0
        for tf_id, i_factor_name, selex_exp_ids in self.find_all_selex_experiments():
            if i_factor_name == factor_name:
                for exp_id in selex_exp_ids:
                    self.add_selex_experiment(exp_id)
                    exps_added += 1
        if exps_added == 0:
            print "Couldnt add for", factor_name
        #assert exps_added > 0
        return

    def add_all_selex_experiments(self):
        for tf_id, factor_name, selex_exp_ids in self.find_all_selex_experiments():
            for selex_exp_id in selex_exp_ids:
                try: 
                    self.add_selex_experiment(selex_exp_id)
                except SeqsTooShort, inst:
                    print inst
                    continue
                except NoBackgroundSequences, inst:
                    print inst
                    continue
                except Exception, inst:
                    raise
    
    @property
    def factor_names(self):
        return [exp.factor_name for exp in self]

    def __iter__(self):
        for experiment in self.experiments:
            yield experiment
        return

    def __init__(self, n_samples, validation_split=0.1):
        self.n_samples = n_samples
        self.validation_split = 0.1
        self.experiments = []

    def __len__(self):
        return len(self.experiments)
