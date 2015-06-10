import os, sys
import subprocess

from ENCODE_ChIPseq_tools import find_chipseq_experiments, find_bams

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Download ENCODE ChIP-seq experiments.')

    parser.add_argument( '--assembly',  required=True, choices=['hg19', 'mm9'],
        help='Which genome and assembly to target.')

    parser.add_argument( '--factors', nargs='*',
        help='Factors to download - use --list-factors to list avaialble factors')

    parser.add_argument( '--list-factors', default=False, action='store_true',
        help='List available factors and exit')

    parser.add_argument( '--samples', nargs='*',
        help='Sample to download - use --list-samples to list available samples')

    parser.add_argument( '--list-samples', default=False, action='store_true',
        help='List available samples and exit')

    parser.add_argument( '--download-and-index', 
                         default=False, action='store_true',
                         help='Download and index the bam files.')
    
    args = parser.parse_args()
    assert not (args.list_factors and args.list_samples), \
            "Must choose either list-factors or list-samples"

    return args



def main():
    args = parse_arguments()
    factors = None if args.factors == None else set(args.factors) 
    samples = None if args.samples == None else set(args.samples) 
    experiments = find_chipseq_experiments(args.assembly)
    ps = []
    for exp_id, exp_meta_data in experiments:
        biosample_type = exp_meta_data['biosample_term_name']
        factor = exp_meta_data['target']['label']
        if samples != None and biosample_type not in samples: continue
        if factors != None and factor not in factors: continue
        if args.list_factors: 
            print factor
        elif args.list_samples:
            print biosample_type
        else:
            print biosample_type, factor, exp_id
            for bam in find_bams(exp_id):
                fname = bam.build_fname()
                if args.download_and_index:
                    cmd = "wget --quiet {} -O {} && samtools index {}".format( 
                        bam.file_loc, fname, fname)
                    p = subprocess.Popen(cmd, shell=True)
                    ps.append(p)
    for p in ps:
        p.wait()

if __name__ == '__main__':
    main()
