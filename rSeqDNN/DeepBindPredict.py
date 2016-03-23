import os
import argparse
import numpy as np
import subprocess
import tempfile
import time

from itertools import izip

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc)
from pysam import FastaFile

from grit.lib.multiprocessing_utils import ThreadSafeFile, fork_and_wait

from pyTFbindtools.peaks import (
    load_labeled_peaks_from_beds, 
    getFileHandle, 
    load_chromatin_accessible_peaks_and_chipseq_labels_from_DB,
    PeaksAndLabelsThreadSafeIterator
)

from rSeqDNN import init_prediction_script_argument_parser

from pyTFbindtools.cross_validation import (
    ClassificationResults, ClassificationResult)

model_id_dict = {}
model_id_dict['T011266_1.02'] = 'D00504.005' # MAX
model_id_dict['T044268_1.02'] = 'D00328.018' # CTCF
model_id_dict['T007405_1.02'] = 'D00736.002' # ARID3A
model_id_dict['T025301_1.02'] = 'D00739.001' # ATF3
model_id_dict['T044305_1.02'] = 'D00742.002' # BCL11A
model_id_dict['T153674_1.02'] = 'D00743.001' # BCL3
model_id_dict['T153671_1.02'] = 'D00744.001' # BCLAF1
model_id_dict['T014206_1.02'] = 'D00746.004' # BHLHE40
model_id_dict['T077335_1.02'] = 'D00747.001' # BRCA1
model_id_dict['T025313_1.02'] = 'D00317.009' # CEBPB
model_id_dict['T076679_1.02'] = 'D00754.003' # E2F6
model_id_dict['T044306_1.02'] = 'D00351.009' # EGR1
model_id_dict['T077988_1.02'] = 'D00356.010' # ELF1
model_id_dict['T077991_1.02'] = 'D00378.009' # ETS1
model_id_dict['T081595_1.02'] = 'D00761.001' # FOXA1
model_id_dict['T077997_1.02'] = 'D00409.004' # GABPA
model_id_dict['T084684_1.02'] = 'D00766.002' # GATA2
model_id_dict['T132746_1.02'] = 'D00446.009' # HNF4A
model_id_dict['T025316_1.02'] = 'D00777.002' # JUN
model_id_dict['T025286_1.02'] = 'D00776.005' # JUND
model_id_dict['T025320_1.02'] = 'D00501.004' # MAFF
model_id_dict['T025323_1.02'] = 'D00503.014' # MAFK
model_id_dict['T014210_1.02'] = 'D00785.001' # MYC
model_id_dict['T014191_1.02'] = 'D00784.004' # MXI1
model_id_dict['T093268_1.02'] = 'D00786.001' # NANOG
model_id_dict['T153691_1.02'] = 'D00789.003' # NFYB
model_id_dict['T153683_1.02'] = 'D00559.006' # NRF1
model_id_dict['T044249_1.02'] = 'D00799.001' # REST
model_id_dict['T138768_1.02'] = 'D00619.007' # RFX5
model_id_dict['T153703_1.02'] = 'D00804.002' # SIN3A
model_id_dict['T093385_1.02'] = 'D00806.003' # SIX5
model_id_dict['T044652_1.02'] = 'D00650.007' # SP1
model_id_dict['T044474_1.02'] = 'D00809.002' # SP2
model_id_dict['T077981_1.02'] = 'D00655.006' # SPI1
model_id_dict['T150542_1.02'] = 'D00817.001' # TBP
model_id_dict['T014212_1.02'] = 'D00818.003' # TCF12
model_id_dict['T144100_1.02'] = 'D00819.002' # TCF7L2
model_id_dict['T151689_1.02'] = 'D00679.004' # TEAD4
model_id_dict['T014218_1.02'] = 'D00700.006' # USF1
model_id_dict['T014176_1.02'] = 'D00823.002' # USF2
model_id_dict['T044261_1.02'] = 'D00710.007' # YY1
model_id_dict['T044578_1.02'] = 'D00825.001' # ZBTB33
model_id_dict['T044594_1.02'] = 'D00714.004' # ZBTB7A
model_id_dict['T044468_1.02'] = 'D00723.006' # ZNF143

def get_deepbind_model_id(tf_id):
    return model_id_dict[tf_id]

def get_probability_from_score(score):
    '''pass scores through sigmoid
    '''    
    return 1. / (1. + np.exp(-1.*score))


def load_predictions(fname):
    peaks = []
    prbs = []
    labels = []
    with open(fname) as fp:
        for line in fp:
            peak, label, score = line.split()
            peaks.append(peak)
            prbs.append(get_probability_from_score(float(score)))
            labels.append(int(label))
    return peaks, np.array(prbs), np.array(labels)

def score_regions_with_deepbind(
        peaks_and_labels_iterator, ofp, tf_id, genome_fasta_fname ):
    # make sure the input objects are thread safe
    assert isinstance(
        peaks_and_labels_iterator, PeaksAndLabelsThreadSafeIterator)
    assert isinstance(
        ofp, ThreadSafeFile)
    genome_fasta = FastaFile(genome_fasta_fname)
    
    # get the deep bind model id for this tf_id
    model_id = get_deepbind_model_id(tf_id)

    # get a temporary file to write to
    temp_ofp = tempfile.TemporaryFile()
    
    # spawn a deepbind process
    deepbind_process = subprocess.Popen(
        " ".join(('deepbind', ' --no-head', model_id)),
        stdout=temp_ofp, 
        stdin=subprocess.PIPE,
        shell=True)
    
    # write the peaks sequence to a fasta file
    peaks = []
    labels = []
    try:
        for pk, sample, label, score in peaks_and_labels_iterator:
            if peaks_and_labels_iterator._cur_val%10000 == 0:
                print peaks_and_labels_iterator._cur_val, peaks_and_labels_iterator.n
            peaks.append((pk, sample))
            labels.append(label)
            seq = genome_fasta.fetch(pk.contig, pk.start, pk.stop)
            deepbind_process.stdin.write(seq + "\n")
        deepbind_process.stdin.close()
        deepbind_process.wait()
    except Exception as e:
        if e.errno==32:
            print 'DeepBind closed unexpectedly!'
        raise e

    temp_ofp.seek(0)
    for (peak, sample), line, label in izip(peaks, temp_ofp, labels):
        ofp.write("%s_%s_%i_%i\t%i\t%f\n" % (
            sample, peak.contig, peak.start, peak.stop, label, float(line)))
    temp_ofp.close()
    return

def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for testing rSeqDNN')
    parser.add_argument('--validation-contigs', type=str, default=None,
                    help='to validate on chr1 and chr4, input chr1,chr4')
    args = parser.parse_args()

    if args.half_peak_width >= 500:
        raise ValueError('DeepBind requires half peak width less than 500!')
    if args.validation_contigs is not None:
        args.validation_contigs = set(args.validation_contigs.split(','))
    assert args.annotation_id is not None or args.genome_fasta is not None, \
        "Must set either --annotation-id or --genome-fasta"
    if args.genome_fasta is None:
        assert args.annotation_id is not None
        from pyTFbindtools.DB import load_genome_metadata
        genome_fasta = FastaFile(
            load_genome_metadata(args.annotation_id).filename) 
    else:
        genome_fasta = args.genome_fasta
    
    if args.tf_id is not None:
        assert args.pos_regions is None and args.neg_regions is None, \
            "It doesnt make sense to set both --tf-id and either --pos-regions or --neg-regions"
        assert args.annotation_id is not None, \
            "--annotation-id must be set if --tf-id is set"
        assert args.genome_fasta is None, \
            "if --tf-id is set the genome fasta must be specified by the --annotation-id"

        peaks_and_labels = load_chromatin_accessible_peaks_and_chipseq_labels_from_DB(
            args.tf_id,
            args.annotation_id,
            args.half_peak_width, 
            args.max_num_peaks_per_sample,
            include_ambiguous_peaks=True)
    else:
        assert args.pos_regions != None and args.neg_regions != None, \
            "either --tf-id or both (--pos-regions and --neg-regions) must be set"
        peaks_and_labels = load_labeled_peaks_from_beds(
            args.pos_regions, args.neg_regions, args.half_peak_width)
        
    return peaks_and_labels, genome_fasta, args.tf_id, args.threads, args.validation_contigs

def main():
    peaks_and_labels, genome_fasta, tf_id, num_threads, validation_contigs = parse_args()
    results = ClassificationResults()
    for fold_i, (train, valid) in enumerate(
            peaks_and_labels.iter_train_validation_subsets(
                validation_contigs=validation_contigs)):
        valid = valid.remove_ambiguous_labeled_entries()
        print "sample: ", valid.sample_ids
        print "contigs: ", valid.contigs
        peaks_and_labels_iterator = valid.thread_safe_iter()
        ofname = "scores.%s.fold%i.txt" % (tf_id, fold_i)
        ofp = ThreadSafeFile(ofname, "w")
        args = [peaks_and_labels_iterator, ofp, tf_id,genome_fasta.filename]
        fork_and_wait(num_threads, score_regions_with_deepbind, args)
        ofp.close()
        peaks, prbs, labels = load_predictions(ofname)
        # XXX Untested 
        pred_labels = np.zeros(len(prbs))
        pred_labels[prbs > 0.5] = 1.0
        pred_labels[prbs <= 0.5] = 0.0
        res = ClassificationResult(labels, pred_labels, prbs)
        results.append(res)
        print "Fold_%i" % fold_i, res
    print results

if __name__ == '__main__':
    main()
