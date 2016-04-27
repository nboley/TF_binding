import os
import math
import gzip
from collections import namedtuple, defaultdict, OrderedDict
from itertools import izip, chain
import random
import cPickle as pickle
import hashlib

import numpy as np


from sklearn.cross_validation import StratifiedKFold

from pysam import Tabixfile, FastaFile

from pyDNAbinding.sequence import one_hot_encode_sequence

from grit.lib.multiprocessing_utils import Counter
from grit.files.reads import ChIPSeqReads, DNASESeqReads, MergedReads

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
            scores.append(score)
        else:
            labels.append(0)
            scores.append(0)
    return labels, scores

def build_peaks_and_chipseq_labels_matrix(
        regions,
        chipseq_peaks_fnames,
        labeling_callback=label_and_score_peak_with_chipseq_peaks):
    print "Building labels - this could take a while"
    tf_ids = []
    labels = np.zeros((len(pks), len(peak_fnames)), dtype='float32')
    for tf_index, (tf_id, factor_peak_filenames) in enumerate(
            sorted(chipseq_peaks_fnames.iteritems())):
        tf_ids.append(tf_id)
        for i, pk in enumerate(pks):
            pk_labels, pk_scores = label_and_score_peak_with_chipseq_peaks(
                factor_peak_filenames, pk)
            try: label = max(pk_labels)
            except ValueError: label = 0 # if there are no overlapping peaks
            labels[i, tf_index] = label
            if i>0 and i%100000 == 0:
                print "%i/%i peaks, %i/%i samples" % (
                    i, len(pks), tf_index, len(peak_fnames))
    return tf_ids, labels

def load_or_build_peaks_and_labels_mat_from_DB(
        annotation_id, roadmap_sample_id, half_peak_width=None):
    # load all of the peak file names from the database
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, roadmap_sample_id=roadmap_sample_id)[roadmap_sample_id]
    assert not any(x is None for x in peak_fnames.keys())

    # extract the accessibility peak filenames
    dnase_peaks_fnames = peak_fnames['dnase']
    if len(dnase_peaks_fnames) == 0:
        raise ValueError, "Can not load DNASE peaks for roadmap sample ID '%s'" % roadmap_sample_id
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

    # build label the DNASE peaks
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
            chipseq_peaks_fnames = [
                fname for pk_type, fname in chipseq_peaks_fnames
                if pk_type == desired_peak_type
            ]
            tf_ids, labels = build_DNASE_peaks_and_labels_matrix_from_DB(
                annotation_id, roadmap_sample_id, half_peak_width)
            with open(pickle_fname + ".tfnames", "w") as ofp:
                for tf_name in tf_ids:
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


def load_accessibility_data(sample_id, pks):
    import sys
    sys.path.insert(0, "/users/nboley/src/bigWigFeaturize/")
    import bigWigFeaturize
    Region = namedtuple('Region', ['chrom', 'start', 'stop'])
    fname = '/mnt/lab_data/kundaje/jisraeli/DNase/unsmoothed_converage/bigwigs/{}-DNase.bw'.format(sample_id)
    pk_width = pks[0]['stop'] - pks[0]['start']
    cached_fname = "cachedaccessibility.%s.%s.obj" % (
        hashlib.sha1(pks.view(np.uint8)).hexdigest(),
        hash(tuple(fname))
     )
    try:
        raise IOError, 'DONT CACHE'
        with open(cached_fname) as fp:
            print "Loading cached accessibility data"
            rv = np.load(fp)
    except IOError:
        rv = bigWigFeaturize.new(
            [fname,],
            pk_width, 
            intervals=[
                Region(pk['contig'], pk['start'], pk['stop']) for pk in pks
            ]
        )[:,0,0,:]

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
            DNASESeqReads(fname).init() for fname in fnames])
        for i, pk in enumerate(peaks):
            cov = reads.build_read_coverage_array(
                pk[0], '.', pk[1], pk[2]-1)
            rv[0,i,:] = cov
            if i%1000 == 0: print i, len(peaks), cov.shape
        with open(cached_fname, "w") as ofp:
            print "Saving cached DNASE coverage for %s (%s)" % (
                sample_id, cached_fname)
            np.save(ofp, rv)
    return rv[0,:,:]

def one_hot_encode_peaks_sequence(pks, genome_fasta, cache_seqs=True):
    if cache_seqs is True:
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

    if cache_seqs is True:
        with open(cached_fname, "w") as ofp:
            print "Saving seqs"
            np.save(ofp, rv)

    return rv

class Data(object):
    """Store and iterate through data from a deep learning model.

    """
    def save(self):
        """Save the data into an h5 file.

        """
        raise NotImplementedError, "Not implemented"

    @staticmethod
    def load(fname):
        """Load data from an h5 file.

        """
        raise NotImplementedError, "Not implemented"

    def __init__(self, inputs, outputs, task_ids=None):
        # if inputs is an array, then we assume that this is a sequential model
        if isinstance(inputs, np.ndarray):
            if not isinstance(outputs, np.ndarray):
                raise ValueError, "If the input is a numpy array, then the output is also expected to be a numpy array.\nHint: You can use multiple inputs and outputs by passing a dictionary keyed by the dtaa type name."
            self._data_type = "sequential"
            self.num_observations = inputs.shape[0]
            assert self.num_observations == output.shape[0]
            # if no task ids are set, set them to indices
            if task_ids is not None:
                assert len(task_ids) == outputs.shape[1]
            else:
                task_ids = [str(x) for x in xrange(1, outputs.shape[1]+1)]
        # otherwise assume that this is a graph type model
        else:
            self.num_observations = inputs.values()[0].shape[0]
            # make sure that all of that data are arrays and have the same
            # number of observations
            for key, val in chain(inputs.iteritems(), outputs.iteritems()):
                assert isinstance(val, np.ndarray)
                assert self.num_observations == val.shape[0], \
                    "The number of observations ({}) is not equal to the first shape dimension of {} ({})".format(
                        self.num_observations, key, val.shape[0])
            # make sure that task id length match up. If a task id key doesnt
            # exist, then default to sequential numbers
            for key in outputs.iterkeys():
                if key in task_ids:
                    if len(task_ids[key]) != outputs[key].shape[1]:
                        raise ValueError, "The number of task ids for key '{}' does not match the output shape".format()
                pass            
            self._data_type = "graph"

        self.task_ids = task_ids
        self.inputs = inputs
        self.outputs = outputs
    
    def build_label_balanced_indices(self, label_key=None, task_id=None):
        # figure out the labels to sample from
        if self._data_type == "sequential":
            assert task_id is None
            labels = self.outputs
        else:
            if label_name is None and len(self.outputs) == 1:
                label_name = next(self.inputs.iterkeys())
            else:
                assert task_id is not None
                labels = self.inputs[task_id]

        if labels.shape[1] == 1:
            assert (task_id is None) or (task_id in self.task_ids), \
                "The specified task_id ({}) does not exist".format(task_id)
            labels = labels[:,0]
        else:
            assert task_id is not None, "Must choose the task id to balance on in the multi-task setting"
            labels = labels[:,self.task_ids.index(task_id)]

        one_indices = np.random.choice(
            np.nonzero(labels == 1)[0], size=len(labels)/2)
        zero_indices = np.random.choice(
            np.nonzero(labels == 0)[0], size=len(labels)/2)
        permutation = np.concatenate((one_indices, zero_indices), axis=0)
        np.random.shuffle(permutation)
        return permutation

    def build_shuffled_indices(self):
        return np.random.permutation(self.num_observations)

    def build_ordered_indices(self):
        return np.arange(self.num_observations)

    def iter_batches_from_indices_generator(
            self, batch_size, repeat_forever, indices_generator):
        i = 0
        n = int(math.ceil(self.num_observations/float(batch_size)))
        permutation = None
        while repeat_forever is True or i<n:
            if i%n == 0:
                permutation = indices_generator()
            # yield a subset of the data
            subset = slice((i%n)*batch_size, (i%n+1)*batch_size)
            indices = permutation[subset]
            if self._data_type == 'sequential':
                rv = (self.inputs[indices], self.outputs[indices])
            else:
                rv = {}
                for key, val in self.inputs.iteritems():
                    rv[key] = val
                for key, val in self.outputs.iteritems():
                    rv[key] = val                
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
        
        return self.iter_batches_from_indices_generator(
            batch_size, 
             repeat_forever, 
            indices_generator,
            **kwargs
        )

    def subset_observations(self, observation_indices=slice(None)):
        """Return a copy containing only the observations specified by indices

        indices: numpy array of indices to select
        """
        if self._data_type == 'sequential':
            assert isinstance,(self.inputs, np.ndarray)
            new_inputs = self.inputs[observation_indices]
            assert isinstance,(self.outputs, np.ndarray)
            new_outputs = self.outputs[observation_indices]
        elif self._data_type == 'graph':
            new_inputs = {}
            for key, data in self.inputs.iteritems():
                assert isinstance(data, np.ndarray)
                new_inputs[key] = data[observation_indices]
            new_outputs = {}
            for key, data in self.outputs.iteritems():
                assert isinstance(data, np.ndarray)
                new_outputs[key] = data[observation_indices]
        else:
            assert False,"Unrecognized model type '{}'".format(self._data_type)

        rv = Data(new_inputs, new_outputs, self.task_ids)
        rv.__class__ = self.__class__
        return rv
    
    def balance_data(self, task_id=None):
        indices = self.build_label_balanced_indices(task_id=task_id)
        return self.subset_observations(indices)
    
    def subset_tasks(self, desired_task_ids):
        """Return a copy of self that only contains desired_task_ids.

        """
        # make chained filtering more convenient by defaulting to 
        # all tfs if none is passed
        if desired_task_ids is None:
            return self
        
        new_outputs = {}
        for task_key, data in self.outputs.iteritems():
            task_indices = []
            for task_id in desired_task_ids[task_key]:
                try: task_indices.append(self.task_ids[task_key].index(task_id))
                except ValueError: task_indices.append(-1)
            task_indices = np.array(task_indices)

            new_data = np.insert(data, 0, -1, axis=1)
            new_outputs[task_key] = new_data[:, task_indices+1]

        rv = Data(self.inputs, new_outputs, task_ids=desired_task_ids)
        rv.__class__ = self.__class__
        return rv

class GenomicRegionsAndLabels(Data):
    """Subclass Data to handfle the common case where the input is a set of 
       genomic regions and the output is a single labels matrix. 
    
    """
    
    @property
    def label_ids(self):
        return self.task_ids['labels']

    @property
    def regions(self):
        return self.inputs['regions']

    @property
    def labels(self):
        return self.outputs['labels']

    def subset_pks_by_rank(self, max_num_peaks, use_top_accessible, seed=0):
        """Return a copy of self containing at most max_num_peaks peaks.

        max_num_peaks: the maximum number of peaks to return
        use_top_accessible: return max_num_peaks most accessible peaks
        """
        if max_num_peaks is None:
            max_num_peaks = len(self.regions)
        
        # sort the peaks by accessibility
        if use_top_accessible:
            indices = np.lexsort(
                (self.regions['start'], -self.regions['signalValue'])
            )
        # sort the peaks randomly
        else:
            # set a seed so we can use cached peaks between debug rounds
            np.random.seed(seed)
            indices = np.argsort(np.random.random(len(self.regions)))
        return self.subset_observations(indices[:max_num_peaks])
    
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
            i for i, pk in enumerate(self.regions) 
            if (contigs_to_exclude is None or pk[0] not in contigs_to_exclude)
            and (contigs_to_include is None or pk[0] in contigs_to_include)
        ])
        return self.subset_observations(indices)
    
    @property
    def n_samples(self):
        return len(self.regions)

    def __init__(self, regions, labels, inputs={}, task_ids=None):
        # add regions to the input
        if 'regions' in inputs:
            raise ValueError, "'regions' input is passed as an argument and also specified in inputs"
        inputs['regions'] = regions
        Data.__init__(self, inputs, {'labels': labels}, {'labels': task_ids})

class GenomicRegionsAndChIPSeqLabels(GenomicRegionsAndLabels):
    @property
    def factor_names(self):
        if self._factor_names is None:
            from pyTFbindtools.DB import load_tf_names
            self._factor_names = [
                x.replace("eGFP-", "") for x in load_tf_names(self.tf_ids)]
            assert len(self.tf_ids) == len(self._factor_names)
        return self._factor_names

    @property
    def tf_ids(self):
        return self.task_ids
    
    @property
    def seq_length(self):
        return self.inputs['regions'][0][2] - self.inputs['regions'][0][1]
    
    def subset_tfs(self, tf_ids):
        return self.subset_tasks({'labels': tf_ids})

def load_chipseq_data(
        roadmap_sample_id,
        factor_names,
        max_n_samples,
        annotation_id,
        half_peak_width):
    # XXX search for cached data
    (pks, tf_ids, (idr_optimal_labels, relaxed_labels)
        ) = load_or_build_peaks_and_labels_mat_from_DB(
            annotation_id, roadmap_sample_id, half_peak_width)
    print "Coding peaks"
    from pyDNAbinding.DB import load_genome_metadata
    genome_fasta = FastaFile("hg19.genome.fa")
    #load_genome_metadata(annotation_id).filename)
    fwd_seqs = one_hot_encode_peaks_sequence(pks, genome_fasta)
    dnase_cov = load_accessibility_data(roadmap_sample_id, pks)

    # set the ambiguous labels to -1
    ambiguous_pks_mask = (
        (idr_optimal_labels < -0.5)
        | (relaxed_labels < -0.5)
        | (idr_optimal_labels != relaxed_labels)
    )
    idr_optimal_labels[ambiguous_pks_mask] = -1
    labels = idr_optimal_labels

    print "Filtering Peaks"    
    data = GenomicRegionsAndChIPSeqLabels(
        regions=pks,
        labels=labels,
        inputs={'fwd_seqs': fwd_seqs, 'dnase_cov': dnase_cov[:,None,None,:]},
        task_ids=tf_ids
    )
    data = data.subset_pks_by_rank(
        max_num_peaks=max_n_samples, use_top_accessible=False
    )
    data = data.subset_tfs(factor_names)
    return data

class PartitionedSamplePeaksAndChIPSeqLabels():
    def cache_key(self, sample_id):
        return hashlib.sha1(str((
            sample_id, 
            tuple(sorted(self.factor_names)), 
            self.n_samples, 
            self.annotation_id, 
            self.half_peak_width
        ))).hexdigest()

    @staticmethod
    def _load_cached(cached_fname):
        raise NotImplementedError, "Still ned to do this"
        pass
    
    def _save_cached(self):
        raise NotImplementedError, "working..."
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

    def __init__(self, 
                 roadmap_sample_ids, 
                 factor_names, 
                 max_n_samples=None,
                 validation_sample_ids=None,
                 annotation_id=1, 
                 half_peak_width=500):
        # make sure that validation sample id is loaded
        if validation_sample_ids is not None:
            for sample_id in validation_sample_ids: 
                assert sample_id in roadmap_sample_ids

        self.sample_ids = roadmap_sample_ids
        self.validation_sample_ids = validation_sample_ids
        if factor_names is None:
            factor_names = set()
            for x in map(load_tf_names_for_sample, roadmap_sample_ids):
                factor_names.update(x)
        self.factor_names = sorted(factor_names)
        self.n_samples = max_n_samples
        self.annotation_id = annotation_id
        self.half_peak_width = half_peak_width
        self.seq_length = 2*half_peak_width

        self.data = {}
        self.train = {}
        self.validation = {}
        
        for sample_id in self.sample_ids:
            self.data[sample_id] = load_chipseq_data(
                sample_id,
                factor_names,
                max_n_samples,
                annotation_id,
                half_peak_width)
                                
            assert self.data[sample_id].seq_length == self.seq_length
            print "Splitting out train data"        
            if ( validation_sample_ids is None 
                 or sample_id not in validation_sample_ids ):
                self.train[sample_id] = self.data[
                    sample_id].subset_pks_by_contig(
                        contigs_to_exclude=('chr1', 'chr2', 'chr8', 'chr9')
                    )
                
            print "Splitting out validation data"        
            if ( validation_sample_ids is None 
                 or sample_id in validation_sample_ids ):
                self.validation[sample_id] = self.data[
                    sample_id].subset_pks_by_contig(
                        contigs_to_include=('chr8', 'chr9')
                    )

    def iter_batches( # sample_balanced
            self, batch_size, data_subset, repeat_forever, **kwargs):
        if data_subset == 'train':
            data_subset = self.train
        elif data_subset == 'validation':
            data_subset = self.validation
        else:
            raise ValueError, "Unrecognized data_subset type '%s'" % data_subset

        ## find the number of observations to sample from each batch
        # To make this work, I would need to randomly choose the extra observations
        assert batch_size >= len(data_subset), \
            "Cant have a batch size smaller than the number of samples"
        fractions = np.array([x.n_samples for x in data_subset.values()], dtype=float)
        fractions = fractions/fractions.sum()
        inner_batch_sizes = np.array(batch_size*fractions, dtype=int)
        # accounting for any rounding from the previous step 
        for i in xrange(batch_size - inner_batch_sizes.sum()):
            inner_batch_sizes[i] += 1

        iterators = OrderedDict(
            (sample_id, 
             data.iter_batches(
                 i_batch_size, repeat_forever, **kwargs) )
            for i_batch_size, (sample_id, data) in zip(
                    inner_batch_sizes, data_subset.iteritems())
        )

        def f():
            while True:
                grpd_res = defaultdict(list)
                cnts = []
                for sample_id in self.sample_ids:
                    if sample_id not in iterators:
                        cnts.append(0)
                    elif sample_id in iterators:
                        iterator = iterators[sample_id]
                        data = next(iterator)
                        cnt = None
                        for key, vals in data.iteritems():
                            grpd_res[key].append(vals)
                            if cnt == None: cnt = vals.shape[0]
                            assert cnt == vals.shape[0]
                        cnts.append(cnt)
                    else:
                        assert False
                
                for key, vals in grpd_res.iteritems():
                    grpd_res[key] = np.concatenate(grpd_res[key], axis=0)
                    
                # build the sample labels
                cnts = np.array(cnts)
                sample_labels = np.zeros(
                    (cnts.sum(), len(self.sample_ids)), dtype='float32')
                for i in xrange(len(cnts)):
                    start_index = (0 if i == 0 else np.cumsum(cnts)[i-1])
                    stop_index = np.cumsum(cnts)[i]
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
