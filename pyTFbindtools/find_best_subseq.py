import os, sys

import gzip

sys.path.insert(0, "/users/nboley/src/TF_binding/")
from load_motifs import load_motifs

try: 
    rev_comp_table = str.maketrans("ACGT", "TGCA")
except:
    import string
    rev_comp_table = string.maketrans("ACGT", "TGCA")


def load_selex(fname):
    seqs = []
    with gzip.open(fname) as fp:
        for line_i, line in enumerate(fp):
            if line_i%4 != 1: continue
            #if line[0] in '+@': continue 
            line = line.strip()
            if line == "": continue
            seqs.append( line )
    return seqs

def main():
    motif = load_motifs(sys.argv[1]).values()[0][0]
    for seq in load_selex(sys.argv[2]):
        values = sorted( ((score, i, RC) 
                          for i, RC, score in motif.iter_pwm_score(seq)),
                         reverse=True)
        score, start, RC = values[0]
        subseq = seq[start:start+len(motif)]
        if len(subseq) < len(motif): continue
        if RC: subseq = subseq.translate(rev_comp_table)[::-1] 
        print subseq

main()
