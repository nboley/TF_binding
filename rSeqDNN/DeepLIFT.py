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
    sequence_dl_scores = get_sequence_dl_scores(model.model, pos_sequences)

if __name__ == '__main__':
    args = parse_args()
    main(args)
            
