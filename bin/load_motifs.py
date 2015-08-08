import os, sys

from motif_tools import load_motifs
from collections import defaultdict

# for now we are just using Poyas motifs
MOTIF_DB_FNAME = "MOTIF_DB.txt"

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract motifs.')

    parser.add_argument( '--factors', nargs='+',
        help='Factors to extract: use --list-factors to list available factors')

    parser.add_argument( '--list-factors', default=False, action='store_true',
        help='List available factors and exit')

    parser.add_argument( '--list-motifs', default=False, action='store_true',
        help='List available motifs and exit')

    parser.add_argument( '--write-to-file', default=False, action='store_true',
        help='Write selected motifs to file MOTIFNAME.motif.txt')

    parser.add_argument( '--choose-best-motif', 
                         default=False, action='store_true',
        help='Only choose one motif from each factor')
    
    args = parser.parse_args()
    assert not (args.list_factors and args.list_motifs), \
            "Can not set both --list-factors and --list-motifs"

    return args

def main():
    # missing 'TBP', 'TAF1', 'BCL11A'
    #my_motifs = set(['CTCF', 'POU2F2', 'BATF', 'IRF4', 'REST', 'SPI1',
    #                 'MYC', 'NFKB', 'PAX5', 'TATA', 'TCF12', 'YY1'])
    #print sorted(my_motifs)

    # choose a motif randomly
    grpd_motifs = load_motifs(sys.argv[1])
    for factor, motifs in sorted(grpd_motifs.items()):
        motif = motifs[0]
        with open("%s.motif.txt" % motif.name, "w") as ofp:
            ofp.write(">" + "\n".join(motif.lines) + "\n")

if __name__ == '__main__':
    main()
