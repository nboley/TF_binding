import argparse
import os
import sys
scripts_dir = os.environ.get("UTIL_SCRIPTS_DIR")
if (scripts_dir is None):
    raise Exception("Please set environment variable UTIL_SCRIPTS_DIR")
sys.path.insert(0, scripts_dir)
from synthetic import synthetic

class SimpleSimulations:
    '''
    handles embedders for simple, positional, and density simulations
    '''
    def __init__(self, path_to_motifs, motif_name):
        self.path_to_motifs = path_to_motifs
        self.motif_name = motif_name
        self.loaded_motifs = synthetic.LoadedEncodeMotifs(self.path_to_motifs,
                                                          pseudocountProb=0.001)
    
    def empty_embedder(self):
        return []

    def non_positional_fixed_embedder(self, num_motifs):
        return
        [synthetic.RepeatedEmbedder(
            synthetic.SubstringEmbedder(
                substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
                    loadedMotifs=self.loaded_motifs,motifName=self.motif_name),
                positionGenerator=synthetic.UniformPositionGenerator()),
            quantityGenerator=synthetic.FixedQuantityGenerator(num_motifs))
        ]

    def positional_fixed_embedder(self, num_motifs, central_bp):
        return
        [synthetic.RepeatedEmbedder(
            synthetic.SubstringEmbedder(
                substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
                    loadedMotifs=self.loaded_motifs,motifName=self.motif_name),
                positionGenerator=synthetic.InsideCentralBp(central_bp)),
            quantityGenerator=synthetic.FixedQuantityGenerator(num_motifs))
        ]

    def non_positional_poisson_embedder(self, mean_motifs, max_motifs):
        return ## TODO: support min_motifs
        [synthetic.RepeatedEmbedder(
            synthetic.SubstringEmbedder(
                substringGenerator=synthetic.PwmSamplerFromLoadedMotifs(
                    loadedMotifs=loaded_motifs,motifName=motif_name),
                positionGenerator=synthetic.UniformPositionGenerator()),
            quantityGenerator=synthetic.MinMaxWrapper(
                synthetic.PoissonQuantityGenerator(mean_motifs),
                theMax=max_motifs))
        ]

def get_simulation_embedders(simulation_type, simple_simulation):
    '''get positive and negative embedders
    '''
    if simulation_type=='simple_motif':
        return [simple_simulation.non_positional_fixed_embedder(1),
                simple_simulation.empty_embedder()]
    else:
        raise ValueError('simulation type not supported!')

def generate_peaks_and_labels(positive_embedder, negative_embedder):
    '''generate peaks and labels object
    TODO: 
    unfirom sample
    different contigs for train/valid/test data
    '''
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for testing KerasModel on simulations')
    parser.add_argument('--all-simulations', default=True,
        help='test model on all simulations')
    parser.add_argument("--path-to-motifs", default="motifs.txt",
        help='file with motif matrices')
    parser.add_argument("--motif-name", required=True,
        help='motif name in jaspar list')

    args = parser.parse_args()

    return ( args.all_simulations,
             args.path_to_motifs,
             args.motif_name )

def main():
    all_simulations, path_to_motifs, motif_name  = parse_args()
    simple_simulation = SimpleSimulations(path_to_motifs, motif_name)  
    if all_simulations:
        for simulation_type in ['simple_motif']:
            ( positive_embedder,
              negative_embedder ) = get_simulation_embedders(simulation_type,
                                                             simple_simulation)
            peaks_and_labels = generate_peaks_and_labels(positive_embedder,
                                                         negative_embedder)


if __name__ == '__main__':
    main()
