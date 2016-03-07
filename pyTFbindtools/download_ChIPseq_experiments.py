import os, sys
import subprocess

from ENCODE_ChIPseq_tools import (
    find_ENCODE_chipseq_experiment_ids,
    find_ENCODE_DCC_bams, 
    find_ENCODE_DCC_experiment_metadata
)

from DB import load_ENCODE_target_id, insert_chipseq_bam_into_db, sync_ENCODE_chipseq_bam_files

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
            "Can not set both --list-factors and --list-samples"

    return args

def main():
    args = parse_arguments()
    if args.list_samples is True:
        raise NotImplemented, "List samples is not implemented"
    
    if args.list_factors is True:
        raise NotImplemented, "List factors is not implemented"
    target_ids = None if args.factors == None else set(
        load_ENCODE_target_id(args.factors) )
    samples = None if args.samples == None else set(args.samples) 
    ps = []
    for exp_id in find_ENCODE_chipseq_experiment_ids(args.assembly):
        for file_data in find_ENCODE_DCC_bams(exp_id):
            if samples != None and file_data.sample_type not in samples: 
                continue
            if target_ids != None and file_data.target_id not in target_ids: 
                continue
            print "INSERTING ", file_data
            insert_chipseq_bam_into_db(file_data)

    sync_ENCODE_chipseq_bam_files()
    #for p in ps:
    #    p.wait()

if __name__ == '__main__':
    main()
