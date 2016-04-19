import os
import numpy as np
from pysam import FastaFile

try:
    from rSeqDNN import init_prediction_script_argument_parser
except:
    from __init__ import init_prediction_script_argument_parser

from pyTFbindtools.peaks import load_labeled_peaks_from_beds
from get_signal import encode_peaks_sequence_into_array
from KerasModel import KerasModel, load_model
from ScoreModel import get_sequence_dl_scores


def parse_args():
    parser = init_prediction_script_argument_parser(
        'main script for extracting sequence DeepLIFT scores')
    parser.add_argument('--model-file', required=True, type=str,
        help='pickle/json model architecture file')
    parser.add_argument('--weights-file', required=True, type=str,
        help='model weights file')
    parser.add_argument('--prefix', required=True, type=str,
                        help='path prefix to output files')

    args = parser.parse_args()
    if args.genome_fasta is None:
        assert args.annotation_id is not None
        from pyTFbindtools.DB import load_genome_metadata
        genome_fasta = FastaFile(
            load_genome_metadata(args.annotation_id).filename) 
    else:
        genome_fasta = args.genome_fasta
    if args.pos_regions == None or args.neg_regions == None:
        raise RuntimeError("--pos-regions and --neg-regions must be set")

    return ( args.model_file,
             args.weights_file,
             args.pos_regions,
             args.neg_regions,
             args.max_num_peaks_per_sample,
             genome_fasta,
             args.prefix )

def main(args):
    ( model_fname,
      weights_fname,
      pos_regions,
      neg_regions,
      max_num_peaks_per_sample,
      genome_fasta,
      prefix ) = args
    print("loading model..")
    model = KerasModel(model_fname=model_fname)
    model.model.load_weights(weights_fname)
    half_peak_width = model.model.input_shape[-1]/2
    print("loading regions...")
    peaks_and_labels = load_labeled_peaks_from_beds(
        pos_regions, neg_regions,
        half_peak_width, max_num_peaks_per_sample)
    pos_peaks_and_labels = peaks_and_labels.filter_by_label(1)
    pos_peaks_and_labels = pos_peaks_and_labels.filter_by_contig_edge(
        9000, genome_fasta) # removes peaks in contigs edges
    print("extracting sequence from fasta..")
    pos_sequences = encode_peaks_sequence_into_array(pos_peaks_and_labels.peaks, genome_fasta)
    print("scoring sequences...")
    sequence_dl_scores = get_sequence_dl_scores(model.model, pos_sequences)
    print("writing scores to bedGraph file..")
    lines = []
    sequence_dl_scores_2d = np.sum(sequence_dl_scores.squeeze(), axis=1)
    print "shape of sequence_dl_scores_2d: ", np.shape(sequence_dl_scores_2d)
    print "num of pos peaks: ", len(pos_peaks_and_labels.peaks)
    with open("%s.%s" % (prefix, "bedGraph"), "w") as wf:
        for i, pk in enumerate(pos_peaks_and_labels.peaks):
            chrm = pk.contig
            starts = np.asarray(pk.start + np.arange(pk.pk_width), dtype='str')
            stops = np.asarray(pk.start + np.arange(pk.pk_width) + 1, dtype='str')
            for j in np.arange(pk.pk_width):
                wf.write("%s\t%s\t%s\t%s\n" % (
                    chrm, starts[j], stops[j], str(sequence_dl_scores_2d[i][j])))
    print("processing bedGraph file into bigwig file...")
    os.system("(sort -u -k1,1 -k2,2n -k3,3n -k4,4nr) < %s.bedGraph | (sort -u -k1,1 -k2,2n -k3,3n) > %s.bedGraph.max" % (prefix, prefix))
    os.system("rm %s.bedGraph" % (prefix))
    os.system("bedGraphToBigWig %s.bedGraph.max /mnt/data/annotations/by_organism/human/hg19.GRCh37/hg19.chrom.sizes %s.bw" % (prefix, prefix))
    os.system("rm %s.bedGraph.max" % (prefix))

if __name__ == '__main__':
    args = parse_args()
    main(args)
            
