import os, sys

from motif_tools import iter_motifs
from collections import defaultdict

# for now we are just using Poyas motifs
MOTIF_DB_FNAME = "MOTIF_DB.txt"

def load_motifs(fname, motif_list=None):
    if motif_list != None:
        motif_list = set(x.upper() for x in motif_list)
    obs_factors = set()
    grpd_motifs = defaultdict(list)
    with open(fname) as fp:
        for motif in iter_motifs(fp):
            obs_factors.add(motif.factor)
            if motif_list != None and motif.factor.upper() not in motif_list:
                continue
            grpd_motifs[motif.factor].append(motif)

    for factor, motifs in sorted(grpd_motifs.items()):
        if any(m.meta_data_line.find('jolma') != -1 for m in motifs):
            motifs = [m for m in motifs if m.meta_data_line.find('jolma') != -1]
            for motif in motifs: motif.name += "_selex"
            grpd_motifs[factor] = motifs
            #print factor, 'SELEX'
        elif any(m.meta_data_line.find('bulyk') != -1 for m in motifs):
            motifs = [m for m in motifs if m.meta_data_line.find('bulyk') != -1]
            for motif in motifs: motif.name += "_bulyk"
            grpd_motifs[factor] = motifs
            #print factor, 'BULYK'

        #print factor, len([m.name for m in motifs])
    return grpd_motifs

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
