import os
import math
import gzip
from collections import namedtuple, defaultdict, OrderedDict
from itertools import izip, chain
import random
import cPickle as pickle
import hashlib
import h5py
import tempfile

import numpy as np

from sklearn.cross_validation import StratifiedKFold

from pysam import Tabixfile, FastaFile

import pybedtools

from pyDNAbinding.sequence import one_hot_encode_sequence
from pyDNAbinding.deeplearning import GenomicRegionsAndLabels

from grit.lib.multiprocessing_utils import Counter
from grit.files.reads import ChIPSeqReads, DNASESeqReads, MergedReads

from cross_validation import iter_train_validation_splits

from pyDNAbinding.misc import optional_gzip_open, load_fastq
from pyDNAbinding.deeplearning import Data, SamplePartitionedData

import hashlib
def hashfile(afile, hasher=None, blocksize=65536):
    if hasher is None: hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.hexdigest()

def hash_bed_file(bed_fp):
    with open(bed_fp.fn) as fp:
        return hashfile(fp)

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

def count_num_lines(filename):
    f = getFileHandle(filename)                  
    n_lines = 0
    buf_size = 1024*1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        n_lines += buf.count('\n')
        buf = read_f(buf_size)

    return n_lines

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

def iter_expanded_regions(regions, min_size):
    """Expand regions to cover at least min_size bases.

    """
    for region in regions:
        start, stop = region[1], region[2]
        pk_width = stop-start
        if pk_width < min_size:
            start = max(0, start-(pk_width-min_size)/2)
            stop = start+min_size
        yield NarrowPeak(region[0], start, stop)
    return

def iter_rounded_regions(regions, min_block_size):
    """Expand regions so that region_size.(start/stop)%block_size == 0.

    This is used to ensure that tiled regions dont intersect. 
    """
    for region in regions:
        start, stop = region[1], region[2]
        start = min_block_size*(start//min_block_size)
        if stop%min_block_size > 0:
            stop += (min_block_size - stop%min_block_size)
        yield NarrowPeak(region[0], start, stop)
    return

def build_genomic_regions_array(regions):
    interval_types = ('S64', 'i4', 'i4')
    return np.array(
        [(str(x[0]), int(x[1]), int(x[2])) for x in regions], 
        dtype=zip(('contig', 'start', 'stop'), interval_types)
    )

def build_genomic_regions_array_from_bed(regions):
    return build_genomic_regions_array(regions)

def build_tiled_regions(regions, core_size, flank_size, offset_size):
    """Expand and iterate tiled regions.
    
    """
    assert core_size%offset_size == 0
    new_region_size = core_size + 2*flank_size
    assert new_region_size%offset_size == 0
    regions_str = "\n".join(
        "\t".join(map(str, x[:3])) 
        for x in iter_rounded_regions(
            iter_expanded_regions(regions, new_region_size),
            offset_size
        )
    )
    regions_bed = pybedtools.BedTool(regions_str, from_string=True)
    regions_bed = regions_bed.sort()
    regions_bed = regions_bed.merge()
    tiled_regions = []
    tiled_regions_with_flank = []
    for region in regions_bed:
        region_size = region.stop-region.start
        assert region_size%offset_size == 0
        num_offsets = region_size//offset_size
        for i in xrange(num_offsets):
            start = region.start-flank_size+i*offset_size
            if start < 0: continue
            stop = region.start+core_size+flank_size+i*offset_size
            tiled_regions_with_flank.append(NarrowPeak(region[0], start, stop))
            tiled_regions.append(
                NarrowPeak(region[0], start+flank_size, stop-flank_size))

    regions_str = "\n".join(
        ["\t".join(map(str, x[:3])) for x in tiled_regions])
    regions_bed = pybedtools.BedTool(regions_str, from_string=True)
    regions_bed = regions_bed.sort()

    regions_w_flank_str = "\n".join(
        ["\t".join(map(str, x[:3])) for x in tiled_regions_with_flank])
    regions_w_flank_bed = pybedtools.BedTool(regions_w_flank_str, from_string=True)
    regions_w_flank_bed = regions_w_flank_bed.sort()
    return Regions(
        build_genomic_regions_array_from_bed(regions_bed), 
        build_genomic_regions_array_from_bed(regions_w_flank_bed)
    )

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

def iter_bed_regions(fp, max_n_regions=None):
    for pk in iter_narrow_peaks(fp, max_n_regions):
        yield pk

def load_bed_regions(fp, max_n_regions=None):
    return list(iter_bed_regions(fp, max_n_regions))

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
    pk_types = ('S64', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S64')
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

def load_accessibility_data(fname, regions):
    import sys
    sys.path.insert(0, "/users/nboley/src/bigWigFeaturize/")
    import bigWigFeaturize
    Region = namedtuple('Region', ['chrom', 'start', 'stop'])
    pk_width = regions[0]['stop'] - regions[0]['start']
    cached_fname = "cachedaccessibility.%s.%s.obj" % (
        hashlib.sha1(regions[:].view(np.uint8)).hexdigest(),
        hash(tuple(fname))
     )
    try:
        fp = h5py.File(cached_fname, 'r')
        return fp['data']
    except IOError:
        rv = bigWigFeaturize.new(
            [fname,],
            pk_width, 
            intervals=[
                Region(pk['contig'], pk['start'], pk['stop']) for pk in regions
            ]
        )[:,0,0,:]
        fp = h5py.File(cached_fname, "w")
        rv = fp.create_dataset("data", rv.shape, dtype='float32')    
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
    pk_width = pks[0][2] - pks[0][1]
    shape = (len(pks), 1, 4, pk_width)
    if cache_seqs is True:
        print "Loading cached one-hot encoded sequences...",
        cached_fname = "cachedseqs.%s.h5" % hashlib.sha1(
            pks[:].view(np.uint8)).hexdigest()
        try:
            fp = h5py.File(cached_fname, 'r')
            print "SUCEEDED"
            return fp['seqs']
        except IOError:
            print "FAILED"
            pass

    fp = h5py.File(cached_fname, "w")
    rv = fp.create_dataset("seqs", shape, dtype='float32', fillvalue=0.25)
    for i, data in enumerate(pks):
        if i%100000 == 0: print "Coded %i/%i peaks" % (i, len(pks))
        assert pk_width == data[2] - data[1], str((pk_width, data))
        seq = genome_fasta.fetch(str(data[0]), data[1], data[2])
        if len(seq) != pk_width: continue
        coded_seq = one_hot_encode_sequence(seq)
        rv[i,0,:,:] = coded_seq.swapaxes(0,1)
    fp.flush()
    # flush and close the mmapped file, and then re-open
    return rv

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

def _build_regions_labels_from_beds(
        regions_bed, regions_w_flank_bed, feature_bed):
    # we assume that regions_bed and regions_w_flank_bed are sorted
    feature_bed = feature_bed.sort()
    print "Feature bed", feature_bed.fn
    print "Regions bed", regions_bed.fn
    print "Building labels ...",
    core_labels = []
    for item in regions_bed.intersect(
            b=feature_bed, c=True, f=0.5, F=0.5, e=True):
        core_labels.append(int(item[3]))
    core_labels = np.array(core_labels)

    flank_labels = []
    for item in regions_w_flank_bed.intersect(
            b=feature_bed, c=True, f=1e-12, F=1e-12, e=True):
        flank_labels.append(int(item[3]))
    flank_labels = np.array(flank_labels)

    labels = core_labels.copy()
    labels[np.abs(core_labels - flank_labels) < 1e-6] = -1
    print "FINISHED Building labels"
    return labels

def load_cached_or_build_tiled_genomic_regions(
        regions, core_size, flank_size, offset_size):
    assert isinstance(regions, np.ndarray)
    #regions_array = build_genomic_regions_array(regions)
    regions_sha1_key = hashlib.sha1(regions_array.view(np.uint8)).hexdigest()
    regions_cache_fname = "cachedregions.%s.h5" % regions_sha1_key
    try:
        regions = Regions.load(regions_cache_fname)
    except IOError:
        # Build a bed tools with the sorted regions
        regions = build_tiled_regions(
            regions, core_size, flank_size, offset_size)
        regions.save(regions_cache_fname)
    return regions

def load_cached_or_build_chipseq_labels(regions, peak_fname):
    fp = pybedtools.BedTool(peak_fname)
    data_hash = abs(hash((hash_bed_file(fp), 
                          hash_bed_file(regions.regions_bed), 
                          hash_bed_file(regions.regions_with_flank_bed))))
    cached_fname = "cachedlabels.%s.npy" % data_hash
    try:
        return np.load(cached_fname)
    except IOError:
        labels = _build_regions_labels_from_beds(
            regions.regions_bed, regions.regions_with_flank_bed, fp)
        with open(cached_fname, "w") as ofp: 
            np.save(ofp, labels)
        return labels

def load_regions_genome_wide(genome_fasta_fname, bin_width, stride, flank_size):
    """Returns regions record array.

    """
    raise NotImplementedError

def save_regions_array_to_bedtool(regions):
    bed_fp = tempfile.NamedTemporaryFile("w", dir="./", delete=False, suffix=".bed")
    np.savetxt(bed_fp, regions, fmt="%s\t%i\t%i")
    bed_fp.flush()
    return pybedtools.BedTool(bed_fp.name)

class Regions(object):
    def save(self, ofname):        
        with h5py.File(ofname, "w") as f:
            f['regions'] = self.regions
            f['regions_with_flank'] = self.regions_with_flank
        return

    @classmethod
    def load(cls, fname):
        f = h5py.File(fname, 'r')
        return cls(f['regions'], f['regions_with_flank'])

    def __init__(self, regions, regions_with_flank=None):
        assert isinstance(regions, (np.ndarray, h5py._hl.dataset.Dataset))
        self.regions = regions
        # build a bedtool corresponding to these regions 
        self.regions_bed = save_regions_array_to_bedtool(self.regions)
        # if regions with flanks aren't specified, then set them to the 
        # full region set
        if regions_with_flank is None:
            self.regions_with_flank = self.regions
            self.regions_with_flank_bed = self.regions_bed
        else:
            self.regions_with_flank = regions_with_flank
            self.regions_with_flank_bed = save_regions_array_to_bedtool(
                self.regions_with_flank)
        return

def build_chipseq_data_for_sample(
        regions, 
        genome_fasta_fname,
        chipseq_peaks_fnames, # indexed by tf name
        dnase_signal_fname=None,
        chipseq_relaxed_peaks_fnames=None, # indexed by tf name
        bin_size=400, 
        flank_size=800, 
        offset_size=50
    ):
    """Build a GenomicRegionsAndChIPSeqLabels data object. 

    - one hot encode sequence
    - extracts DNASE signal for region
    - labels peaks
    """
    tiled_regions_cached_fname = "cachedtiledregions.%s.h5" % (
        hashlib.sha1(regions.view(np.uint8)).hexdigest())
    try: 
        print "Loading cached classification region set (core_bin_size: {}   flank_size: {}   offset_size: {})... ".format(
            bin_size, flank_size, offset_size),
        regions = Regions.load(tiled_regions_cached_fname)
    except IOError:
        print " FAILED",
        print "Building classification region set (core_bin_size: {}   flank_size: {}   offset_size: {})".format(
            bin_size, flank_size, offset_size),
        regions = build_tiled_regions(regions, bin_size, flank_size,offset_size)
        regions.save(tiled_regions_cached_fname)
    else:
        print " SUCCEEDED"
    
    tf_ids = sorted(chipseq_peaks_fnames)
    if (chipseq_relaxed_peaks_fnames is not None and
        sorted(chipseq_relaxed_peaks_fnames.iterkeys()) != tf_ids):
        raise ValueError, "The relaxed and optimal peak ids are different."

    # one hot encode sequences
    print "Encoding sequence" 
    genome_fasta = FastaFile(genome_fasta_fname)
    fwd_seqs = one_hot_encode_peaks_sequence(
        regions.regions_with_flank, genome_fasta)

    print "Building ChIP-seq labels"     
    # build the chipseq labels
    labels = {}
    for tf_id, fname in chipseq_peaks_fnames.iteritems():
        labels[tf_id] = load_cached_or_build_chipseq_labels(regions, fname)
        # we've already tested to make sure that the tf ids match
        if chipseq_relaxed_peaks_fnames is not None:
            relaxed_labels = load_cached_or_build_chipseq_labels(
                regions, chipseq_relaxed_peaks_fnames[tf_id])
            ambiguous_pks_mask = (
                (labels[tf_id] < -0.5)
                | (relaxed_labels < -0.5)
                | (labels[tf_id] != relaxed_labels)
            )
            labels[tf_id][ambiguous_pks_mask] = -1
    # concatanate the tf label arrays into a matrix
    labels = np.concatenate([labels[tf_id] for tf_id in tf_ids], axis=0)
    if len(labels.shape) == 1: labels = labels[:,None]

    # extract DNASE signal
    print "Loading accessibility data"
    dnase_cov = load_accessibility_data(
        dnase_signal_fname, regions.regions_with_flank)
    
    # return the object
    return GenomicRegionsAndChIPSeqLabels(
        regions=regions.regions_with_flank,
        labels=labels,
        inputs={'fwd_seqs': fwd_seqs, 'dnase_cov': dnase_cov},
        task_ids=tf_ids
    )

def load_chipseq_data_for_samples(
        # if regions is a dictionary, then we assume that keys are sample_ids
        # and we match regions with sample ids. Otherwise, we use the same
        # regions for all samples
        regions, 
        # fasta fname
        genome_fasta_fname,
        # dictionary of dictionaries indexed by sample_id, then tf_name
        chipseq_peaks_fnames,
        # dictionary indexed by sample_id 
        dnase_signal_fnames=None,
        # dictionary of dictionaries indexed by sample_id, then tf_name
        chipseq_relaxed_peaks_fnames=None
    ):
    raise NotImplementedError
    # find the sample ids
    # assert that there the same
    assert isinstance(regions, Regions)

    data = {}
    print "Building tiled regions"
    regions = build_tiled_regions(
        regions, core_size=400, flank_size=800, offset_size=50)
    # fix regions to make it accept and index if it's not a dict
    for sample_id in sample_ids:
        data[sample_id] = build_chipseq_data_for_sample(
            regions, 
            genome_fasta_fname, 
            chipseq_peaks_fnames[sample_id], 
            (None if dnase_signal_fnames 
             is None else dnase_signal_fname[sample_id]),
            (None if chipseq_relaxed_peaks_fnames is None 
             else chipseq_relaxed_peaks_fnames[sample_id]),
        )
    return SamplePartitionedData(data)

def load_chipseq_data_from_DB(
        roadmap_sample_id,
        factor_names,
        max_n_samples,
        annotation_id,
        bin_size=400,
        flank_size=800,
        offset_size=50):
    cached_fname = "cachedDBchipseq.%i.h5" % abs(hash((
        roadmap_sample_id, 
        tuple(factor_names), 
        max_n_samples, 
        annotation_id,
        bin_size,
        flank_size,
        offset_size)))
    try:
        return GenomicRegionsAndChIPSeqLabels.load(cached_fname)
    except IOError:
        print "Loading cached file failed... building data set"

    # load the DNASE regions and ChIP-seq peak filenames
    print "Loading peak filenames"
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db, load_tf_ids
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, roadmap_sample_id=roadmap_sample_id)[roadmap_sample_id]
    dnase_peaks_fname = list(peak_fnames['dnase'])[0]
    print "Loading base regions"
    with optional_gzip_open(dnase_peaks_fname) as fp:
        regions = build_genomic_regions_array(load_bed_regions(fp))
    # for each tf_id in the desired list, find the peak fnames and then 
    # segregate into relaxed and optimal sets 
    optimal_peaks_fnames = defaultdict(list)
    relaxed_peaks_fnames = defaultdict(list)
    for tf_id in load_tf_ids(factor_names):
        for peak_type, fname in peak_fnames[tf_id]:
            if peak_type == 'optimal idr thresholded peaks':
                optimal_peaks_fnames[tf_id].append(fname)
            else:
                relaxed_peaks_fnames[tf_id].append(fname)

    print "Organizing ChIP-seq peak files"
    # make sure there is only one unique optimal peak per tf_id. If there
    # are multiple, we take the peak file with the most peaks
    for key, fnames in optimal_peaks_fnames.items():
        # if there's more than one file, choose the one with the most of peaks
        if len(fnames) > 1:
            line_cnts = [count_num_lines(fname) for fname in fnames]
            fname = fnames[line_cnts.index(max(line_cnts))]
        else:
            assert len(fnames) == 1
            fname = fnames[0]
        optimal_peaks_fnames[key] = fname
    # convert the default dict into a standard dict
    optimal_peaks_fnames = dict(optimal_peaks_fnames)
    # merge all of the relaxed peaks into a single peak file
    for key, fnames in relaxed_peaks_fnames.items():
        if len(fnames) == 1:
            relaxed_peaks_fnames[key] = fnames[0]
        else:
            assert len(fnames) > 1
            bed_fps = [pybedtools.BedTool(fname).sort() for fname in fnames]
            merged_bed = bed_fps[0].cat(*bed_fps[1:]).sort().merge()
            relaxed_peaks_fnames[key] = merged_bed.fn
    # convert the default dict into a standard dict
    relaxed_peaks_fnames = dict(relaxed_peaks_fnames)
    
    # load the genome fasta filename
    from pyDNAbinding.DB import load_genome_metadata
    genome_fasta_fname = load_genome_metadata(annotation_id).filename

    # load the DNASE signal fname
    dnase_fname = \
        '/mnt/lab_data/kundaje/jisraeli/DNase/unsmoothed_converage/bigwigs/{}-DNase.bw'.format(
            roadmap_sample_id)

    # load the chipseq peak fname
    data = build_chipseq_data_for_sample(
        regions, 
        genome_fasta_fname,
        optimal_peaks_fnames, # indexed by tf name
        dnase_signal_fname=dnase_fname,
        chipseq_relaxed_peaks_fnames=relaxed_peaks_fnames, # indexed by tf name
        bin_size=400, 
        flank_size=800, 
        offset_size=50
    )
    data.save(cached_fname)
    return data

class PartitionedSamplePeaksAndChIPSeqLabels():
    @property
    def cache_key(self):
        return hashlib.sha1(str((
            tuple(sorted(self.sample_ids)), 
            None if self.validation_sample_ids \
                is None else tuple(sorted(self.validation_sample_ids)), 
            tuple(sorted(self.factor_names)), 
            self.n_samples, 
            self.annotation_id, 
            self.half_peak_width
        ))).hexdigest()

    @property
    def cache_fname(self):
        return "cachedTFdata.%s.h5" % self.cache_key
    
    @classmethod
    def load(cls, fname):
        with h5py.File(fname, 'r') as f:
            roadmap_sample_ids = f.attrs['roadmap_sample_ids']
            factor_names = f.attrs['factor_names']
            max_n_samples = f.attrs['max_n_samples']
            if max_n_samples == 'NONE': max_n_samples = None
            validation_sample_ids = f.attrs['validation_sample_ids']
            if validation_sample_ids == 'NONE': validation_sample_ids = None
            annotation_id = f.attrs['annotation_id']
            half_peak_width = f.attrs['half_peak_width']

        return cls(roadmap_sample_ids, 
                   factor_names, 
                   max_n_samples, 
                   validation_sample_ids, 
                   annotation_id, 
                   half_peak_width)

    def _load_cached_data(self, f=None):
        print "Loading cached data from '%s'" % self.cache_fname
        if f is None:
            f = h5py.File(self.cache_fname, 'r')
        
        train_data = {}
        for key, val in f['train'].items():
            train_data[key] = Data(*val.values())
        self.train = SamplePartitionedData(train_data)

        validation_data = {}
        for key, val in f['validation'].items():
            validation_data[key] = Data(*val.values())
        self.validation = SamplePartitionedData(validation_data)
        return
        
    def save(self, fname):
        with h5py.File(fname, "w") as f:
            # save the attributes
            f.attrs['roadmap_sample_ids'] = sorted(self.sample_ids)
            f.attrs['factor_names'] = sorted(self.factor_names)
            f.attrs['max_n_samples'] = (
                'NONE' if self.max_n_samples is None 
                else self.max_n_samples )
            f.attrs['validation_sample_ids'] = (
                'NONE' if self.validation_sample_ids is None else 
                sorted(self.validation_sample_ids)
            )
            f.attrs['annotation_id'] = self.annotation_id
            f.attrs['half_peak_width'] = self.half_peak_width
            
            # link to the data 
            f['train'] = h5py.ExternalLink(self.train.cache_to_disk(), "/")
            f['validation'] = h5py.ExternalLink(self.validation.cache_to_disk(), "/")

        return fname

    def cache_to_disk(self):
        return self.save(self.cache_fname)
    
    def __init__(self, 
                 roadmap_sample_ids, 
                 factor_names, 
                 max_n_samples=None,
                 validation_contig_ids=['chr8', 'chr9'],
                 validation_sample_ids=None,
                 test_contig_ids=['chr1', 'chr2'],
                 annotation_id=1,
                 half_peak_width=500):
        # make sure that validation sample id is loaded
        if validation_sample_ids is not None:
            for sample_id in validation_sample_ids: 
                assert sample_id in roadmap_sample_ids

        self.sample_ids = roadmap_sample_ids
        self.validation_sample_ids = validation_sample_ids
        
        # if we aren't provided tf names, then load them from 
        # the database
        if factor_names is None:
            from pyTFbindtools.DB import load_tf_names_for_sample
            factor_names = set()
            for x in map(load_tf_names_for_sample, roadmap_sample_ids):
                factor_names.update(x)
        self.factor_names = sorted(factor_names)
        
        # set the other meta data
        self.max_n_samples = max_n_samples
        self.n_samples = max_n_samples
        self.annotation_id = annotation_id
        self.half_peak_width = half_peak_width
        self.seq_length = 2*half_peak_width

        try:
            self._load_cached_data()
            return
        except IOError:
            print "Can not find cached filename '%s'. Building data." % self.cache_fname
        
        self.train = {}
        self.validation = {}
        
        for sample_id in self.sample_ids:
            data = load_chipseq_data_from_DB(
                sample_id,
                factor_names,
                max_n_samples,
                annotation_id,
                half_peak_width)
            fname = data.cache_to_disk()
            
            assert data.seq_length == self.seq_length
            print "Splitting out train data"
            if ( validation_sample_ids is None 
                 or sample_id not in validation_sample_ids ):
                self.train[sample_id] = data.subset_pks_by_contig(
                        contigs_to_exclude=(
                            list(test_contig_ids) + list(validation_contig_ids))
                    )
                
            print "Splitting out validation data"        
            if ( validation_sample_ids is None 
                 or sample_id in validation_sample_ids ):
                self.validation[sample_id] = data.subset_pks_by_contig(
                        contigs_to_include=validation_contig_ids
                    )

        self.train = SamplePartitionedData(self.train)
        self.validation = SamplePartitionedData(self.validation)
        
        self.fname = self.cache_to_disk()

def test_read_data():
    data = load_chipseq_data('E123', ['CTCF',], 5000, 1, 500)
    fname = data.cache_to_disk()
    data2 = Data.load(fname)
    for x in data2.iter_batches(10):
        break

def test_rec_array():
    from DB import load_all_chipseq_peaks_and_matching_DNASE_files_from_db
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        1, roadmap_sample_id='E114')['E114']
    fname = next(iter(peak_fnames.values()[0]))[1]
    with optional_gzip_open(fname) as fp:
        pks = list(iter_summit_centered_peaks(
            iter_narrow_peaks(fp), 500))
        pk_record_type = type(pks[0])
        pk_types = ('S64', 'i4', 'i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S64')
        pks = np.array(pks, dtype=zip(pks[0]._fields, pk_types))
    with h5py.File('tmp.h5', "w") as f:
        f.create_dataset('test', data=pks)
        print f['test'][1]
    print fname

def test_load_data_from_db():
    print "Testing loading data."
    rv = load_chipseq_data_from_DB(
        roadmap_sample_id='E123',
        factor_names=['CTCF',],
        max_n_samples=5000,
        annotation_id=1,
        bin_size=400,
        flank_size=800,
        offset_size=50
    )
    #rv = PartitionedSamplePeaksAndChIPSeqLabels(['E123',], ['CTCF',])
    print rv.train
    for x in rv.iter_train_data(10):
        for k, v in x.iteritems():
            print k, v.shape
            print v
        break
    return

def test_build_chipseq_labels():
    annotation_id = 1
    roadmap_sample_id='E123'
    tf_id='T153674_1.02'
    from DB import (
        load_all_chipseq_peaks_and_matching_DNASE_files_from_db, 
        load_genome_metadata )
    peak_fnames = load_all_chipseq_peaks_and_matching_DNASE_files_from_db(
        annotation_id, roadmap_sample_id=roadmap_sample_id)[roadmap_sample_id]
    dnase_peaks_fname = list(peak_fnames['dnase'])[0]
    peak_fnames = [fname for pk_type, fname in peak_fnames[tf_id] 
                      if pk_type == 'optimal idr thresholded peaks']
    with optional_gzip_open(dnase_peaks_fname) as fp:
        regions = load_bed_regions(fp)[:5000]
    print "Building tiled regions"
    regions = build_tiled_regions(
        regions, core_size=400, flank_size=800, offset_size=50)
    genome_fasta_fname = load_genome_metadata(annotation_id).filename
    genome_fasta = FastaFile("hg19.genome.fa")
    fwd_seqs = one_hot_encode_peaks_sequence(
        regions.regions_with_flank, genome_fasta)
    assert False
    bin_size, flank_size, offset_size = 400, 800, 50
    labels = load_cached_or_build_chipseq_labels(
        regions, peak_fnames[0], bin_size, flank_size, offset_size)
    return

def test():
    #test_rec_array()
    #test_read_data()
    #test_load_and_save(np.zeros((10000, 50)), np.zeros((10000, 1)))
    #test_load_and_save(
    #    {'seqs': np.zeros((10000, 50))}, {'labels': np.zeros((10000, 1))})
    #test_sample_partitioned_data()
    #test_hash()
    #test_load_data_from_db()
    #test_build_chipseq_labels()
    pass

if __name__ == '__main__':
    test()
