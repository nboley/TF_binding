import os
import gzip
from collections import namedtuple
from itertools import izip, chain
import random
import cPickle as pickle
import ntpath

import numpy as np
from sklearn.cross_validation import StratifiedKFold

from pysam import TabixFile, tabix_index
from pybedtools import Interval, BedTool

from grit.lib.multiprocessing_utils import Counter

from cross_validation import iter_train_validation_splits

def getFileHandle(filename, mode="r"):
    def getHandle(filename):
        endings = ['.gz', '.gzip', '.bgz']
        endswith_endings = [filename.endswith(ending) for ending in endings]
        if any(endswith_endings):
            return gzip.open(filename, mode="rb")
        else:
            return open(filename, mode="r")
    if ',' in filename:
        fnames = filename.split(',')
        return [getHandle(fname) for fname in fnames]
    else:
        return getHandle(filename)

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

    def outside_contig_edge(self, contig_edge, genome_fasta):
        """
        Checks if peak is within specified distance of contig edge.
        Note: norm window starts from peak center, not summit.
        """
        contig_start_edge = 0 + contig_edge
        contig_stop = genome_fasta.lengths[genome_fasta.references.index(self.contig)]
        contig_end_edge = contig_stop - contig_edge
        if self.start>contig_start_edge and self.stop<=contig_end_edge:
            return True
        else:
            return False

    def bins(self, bin_size):
        """
        Bins NarrowPeak, generates binned NarrowPeaks
        Note: peak sequence is removed
        """
        for bin_start in xrange(self.start, self.stop-bin_size/2, bin_size):
            if bin_start-bin_size >= 0:
                yield NarrowPeak(
                    self.contig, bin_start, bin_start+bin_size,
                    bin_size, self.score, self.signalValue, self.pValue, self.qValue,
                    self.idrValue, None)

    def slop(self, flank_size):
        """
        Add flanks, same as bedtools slop. Removes underlying sequence.
        """
        return NarrowPeak(
                self.contig, self.start-flank_size, self.stop+flank_size,
                self.summit+flank_size, self.score, self.signalValue, self.pValue, self.qValue,
                self.idrValue, None)

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

PeakAndLabelData = namedtuple('PeakAndLabel', ['peak', 'sample', 'label', 'score'])
class PeakAndLabel(PeakAndLabelData):
    def jitter(self, jitter):
        """
        Returns PeakAndLabel with jittered NarrowPeak.
        """
        pk, sample, label, score  = self
        jittered_pk = NarrowPeak(
            pk.contig, pk.start, pk.stop, pk.summit,
            pk.score, pk.signalValue, pk.pValue, pk.qValue, pk.idrValue, pk.seq)

        return PeakAndLabel(jittered_pk, sample, label, score)

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
        '''return subset of data without -1 labels
        '''
        return PeaksAndLabels(
                pk_and_label for pk_and_label in self 
                if pk_and_label.label != -1
            )

    def filter_by_label(self, label):
        """
        Keep subset of data with specified label.
        """
        return PeaksAndLabels(
            pk_and_label for pk_and_label in self
            if pk_and_label.label == label
        )

    def filter_by_contig_edge(self, contig_edge, genome_fasta):
        """
        Removes peaks where norm window runs over chr edge.
        """
        return PeaksAndLabels(
            pk_and_label for pk_and_label in self
            if pk_and_label.peak.outside_contig_edge(contig_edge, genome_fasta)
        )

    def iter_train_validation_subsets(
            self, validation_contigs=None, same_celltype=False, single_celltype=False):
        for train_indices, valid_indices in iter_train_validation_splits(
                self.sample_ids, self.contigs,
                validation_contigs, same_celltype, single_celltype):
            yield (self.subset_data(*train_indices),
                   self.subset_data(*valid_indices))

    def jitter_peaks(self, jitter):
        """
        Jitter peak locations.
        Signal and label attributes remain unchanged.

        Paramters
        ---------
        jitter : int
            Jittering peaks by this distance.

        Returns
        -------
        PeaksAndLabels object with jittered peak locations.
        """
        return PeaksAndLabels(
            pk_and_label.jitter(jitter) for pk_and_label in self
        )

    def slop_peaks(self, flank_size):
        """
        add flanks to peaks.
        """
        return PeaksAndLabels(
            PeakAndLabel(pk.slop(flank_size), sample, label, score)
            for pk, sample, label, score in self
        )

def merge_peaks_and_labels(*peaks_and_labels_iterable):
    """
    Merge multiple PeaksAndLabels into a single PeaksAndLabels.
    """
    return PeaksAndLabels(chain.from_iterable(*peaks_and_labels_iterable))

def get_intervals_from_peaks(peaks):
    '''returns list of pybedtools intervals
    '''
    return [Interval(pk.contig, pk.start, pk.stop) for pk in peaks]

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

def iter_bedtool_peaks(bedtool, max_num_peaks=None):
    idrValue = -1.0
    seq = None
    for i, interval in enumerate(bedtool):
        if max_num_peaks != None and i > max_num_peaks:
            break
        chrm = str(interval.chrom)
        start = int(interval.start)
        stop = int(interval.stop)

        try: summit = int(interval.fields[9])
        except IndexError: summit = (stop-start)/2
        try: score = float(interval.fields[4])
        except IndexError: score = -1.0
        try: signalValue = float(interval.fields[6])
        except IndexError: signalValue = -1.0
        try: pValue = float(interval.fields[7])
        except IndexError: pValue = -1.0
        try: qValue = float(interval.fields[8])
        except IndexError: qValue = -1.0

        yield NarrowPeak(
            chrm, start, stop, summit,
            score, signalValue, pValue, qValue, idrValue, seq)

    return

def merge_peaks(pk_iter):
    """
    uses merge from pybedtools to merge NarrowPeaks
    """
    intervals = get_intervals_from_peaks(pk_iter)
    bedtool = BedTool(intervals)
    merged_bedtool = bedtool.sort().merge()

    return iter_bedtool_peaks(merged_bedtool)

def save_peaks_bedfile(pk_iter, ofname):
    """
    Saves bedfile with peak coordinates.
    """
    intervals = get_intervals_from_peaks(pk_iter)
    bedtool = BedTool(intervals)
    result = bedtool.saveas(ofname)

    return

def load_labeled_peaks_from_beds(
        pos_regions_fp, neg_regions_fp, 
        half_peak_width=None,
        max_num_peaks_per_sample=None):
    def iter_all_pks():        
        for pos_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(pos_regions_fp, max_num_peaks_per_sample), half_peak_width):
            yield PeakAndLabel(pos_pk, 'sample', 1, pos_pk.signalValue)
        for neg_pk in iter_summit_centered_peaks(
                iter_narrow_peaks(neg_regions_fp, max_num_peaks_per_sample), half_peak_width):
            yield PeakAndLabel(neg_pk, 'sample', 0, neg_pk.signalValue)
    return PeaksAndLabels(iter_all_pks())

def load_and_label_peaks_from_beds(
        background_regions_fp, pos_regions_fp_list,
        ambiguous_regions_fp_list=None,
        bin_size=200, flank_size=400,
        max_num_peaks=None, include_pos_regions=True,
        half_peak_width=None):
    """
    Bins background regions, labels with positive regions, adds flanks.

    Parameters
    ----------
    background_regions_fp: filepointer
    pos_regions_fp_list: list of filpointers
    ambiguous_regions_fp_list: list of filepointers

    Returns
    -------
    PeaksAndLabels
    """
    pos_regions_fname_list = [fp.name for fp in pos_regions_fp_list]
    if ambiguous_regions_fp_list is not None:
        assert len(ambiguous_regions_fp_list)==len(pos_regions_fp_list), \
            "number of ambiguous region files must equal number of pos region files!"
        ambiguous_regions_fname_list = [fp.name for fp in ambiguous_regions_fp_list]
    else: ambiguous_regions_fname_list = None
    sample_name = ntpath.basename(background_regions_fp.name)
    pk_iter = iter_narrow_peaks(background_regions_fp, max_num_peaks)
    if half_peak_width is not None:
        pk_iter = iter_summit_centered_peaks(pk_iter, half_peak_width)
    if include_pos_regions:
        pk_iter_list = [pk_iter] + [iter_narrow_peaks(regions_fp, max_num_peaks)
                                    for regions_fp in pos_regions_fp_list]
        concat_pk_iter = chain.from_iterable(pk_iter_list)
        pk_iter = merge_peaks(concat_pk_iter)
    def iter_pk_and_label(pk_iter, pos_regions_fname_list, ambiguous_regions_fname_list):
        for pk in pk_iter:
            for pk_bin in pk.bins(bin_size):
                labels, scores = label_and_score_peak_with_chipseq_peaks(
                    pos_regions_fname_list, pk_bin)
                labels = np.array(labels)
                scores = np.array(scores)
                if ambiguous_regions_fname_list is not None:
                    ambiguous_labels, ambiguous_scores = label_and_score_peak_with_chipseq_peaks(
                        ambiguous_regions_fname_list, pk_bin)
                    labels_to_ignore = np.array(ambiguous_labels)>labels
                    labels[labels_to_ignore] = -1
                yield PeakAndLabel(
                    pk_bin.slop(flank_size), sample_name, labels, scores)
                    #pk_bin.slop(flank_size), sample_name, labels[0], scores[0])
    return PeaksAndLabels(iter_pk_and_label(
        pk_iter, pos_regions_fname_list, ambiguous_regions_fname_list))

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
            try:
                fp = TabixFile(fname)
            except IOError:
                unzipped_fname = '.'.join(
                    fname.split('.')[:-1]) if fname.endswith('.gz') else fname
                sorted_bed = BedTool(fname).sort().saveas(unzipped_fname)
                tabix_fname = tabix_index(unzipped_fname, preset="bed", force=True)
                fp = TabixFile(tabix_fname)
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

def load_matching_dnase_foldchange_fnames_from_DB(tf_id, annotation_id):
    """
    Checks samples available for target tf,
    and loads dnase filenames for those samples.
    """
    from DB import (
        load_samples_from_db_by_tfid,
        load_DNASE_foldchange_files_from_db_by_sample )
    samples = load_samples_from_db_by_tfid(tf_id, annotation_id)

    return load_DNASE_foldchange_files_from_db_by_sample(samples)

def load_conservation_fnames_from_DB():
    """
    Returns sequence conservation filenames.
    """
    from DB import load_conservation_files_from_db

    return load_conservation_files_from_db()

def load_matching_dnase_cut_fnames_from_DB(tf_id, annotation_id):
    """
    Checks samples available for target tf,
    and loads dnase filenames for those samples.
    """
    from DB import (
        load_samples_from_db_by_tfid,
        load_DNASE_cut_files_from_db_by_sample )
    samples = load_samples_from_db_by_tfid(tf_id, annotation_id)

    return load_DNASE_cut_files_from_db_by_sample(samples)

