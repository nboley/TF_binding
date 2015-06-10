import sys

import math

import numpy
from scipy.optimize import brute, bisect

from collections import defaultdict

T = 300
R = 1.987e-3 # in kCal
#R = 8.314e-3 # in kJ

REG_LEN = 100000

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RC_base_map = {'A': 3, 'C': 2, 'G': 1, 'T': 0}

def logit(x):
    return math.log(x) - math.log(1-x)

def logistic(x):
    try: e_x = math.exp(x)
    except: e_x = numpy.exp(x)
    return e_x/(1+e_x)

def estimate_unbnd_conc_in_region(
        motif, score_cov, atacseq_cov, chipseq_rd_cov,
        frag_len, max_chemical_affinity_change):
    # trim the read coverage to account for the motif length
    trimmed_atacseq_cov = atacseq_cov[len(motif)+1:]
    chipseq_rd_cov = chipseq_rd_cov[len(motif)+1:]

    # normalzie the atacseq read coverage
    atacseq_weights = trimmed_atacseq_cov/trimmed_atacseq_cov.max()
    
    # build the smoothing window
    sm_window = numpy.ones(frag_len, dtype=float)/frag_len
    sm_window = numpy.bartlett(2*frag_len)
    sm_window = sm_window/sm_window.sum()

    def build_occ(log_tf_conc):
        raw_occ = logistic(log_tf_conc + score_cov/(R*T))
        occ = raw_occ*atacseq_weights
        smoothed_occ = numpy.convolve(sm_window, occ/occ.sum(), mode='same')

        return raw_occ, occ, smoothed_occ

    def calc_lhd(log_tf_conc):
        raw_occ, occ, smoothed_occ = build_occ(-log_tf_conc)
        #diff = (100*smoothed_occ - 100*rd_cov/rd_cov.sum())**2
        lhd = -(numpy.log(smoothed_occ + 1e-12)*chipseq_rd_cov).sum()
        #print log_tf_conc, diff.sum()
        return lhd

    res = brute(calc_lhd, ranges=(
        slice(0, max_chemical_affinity_change, 1.0),))[0]
    log_tf_conc = max(0, min(max_chemical_affinity_change, res))
                      
    return -log_tf_conc


class Motif():
    def __len__(self):
        return self.length

    def iter_pwm_score(self, seq):
        seq = seq.upper()
        for offset in xrange(len(seq) - len(self)):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = 0.0
            RC_score = 0.0
            if 'N' in subseq: 
                yield offset + len(self)/2, 0.25*len(self)
                continue
            for i, base in enumerate(subseq):
                score += self.pwm[i][base_map[base]]
                RC_score += self.pwm[len(self)-i-1][RC_base_map[base]]
            yield offset + len(self)/2, max(score, RC_score)

    def iter_seq_score(self, seq):
        seq = seq.upper()
        for offset in xrange(len(seq) - len(self)):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = self.consensus_energy
            RC_score = self.consensus_energy
            if 'N' in subseq:
                yield offset + len(self)/2, self.mean_energy
                continue
            for i, base in enumerate(subseq):
                score += self.motif_data[i][base_map[base]]
                RC_score += self.motif_data[len(self)-i-1][RC_base_map[base]]
            yield offset + len(self)/2, max(score, RC_score)

    def __init__(self, text):
        # load the motif data
        lines = text.strip().split("\n")
        if lines[0][0] == '>': lines[0] = lines[0][1:]
        self.name = lines[0].split()[0]
        self.factor = self.name.split("_")[0]
        
        self.lines = lines
        self.meta_data_line = lines[0]

        self.length = len(lines)-1

        self.consensus_energy = 0.0
        self.motif_data = numpy.zeros((self.length, 4), dtype=float)
        
        self.pwm = numpy.zeros((self.length, 4), dtype=float)
        
        for i, line in enumerate(lines[1:]):
            row = numpy.array([logit(1e-3/2 + (1-1e-3)*float(x)) 
                               for x in line.split()[1:]])
            max_val = row.max()
            self.consensus_energy += max_val
            row -= max_val
            self.motif_data[i, :] = row

            pwm_row = numpy.array([
                float(x) for x in line.split()[1:]])
            self.pwm[i, :] = pwm_row
        
        # reset the consensus energy so that the strongest binder
        # has a binding occupancy of 0.999 at chemical affinity 1
        self.consensus_energy = 12.0/(R*T) #logit(consensus_occupancy)
        consensus_occupancy = logistic(self.consensus_energy)
        
        # calculate the mean binding energy
        mean_energy_diff = sum(row.sum()/4 for row in self.motif_data) 
        def f(scale):
            mean_occ = 1e-100 + logistic(
                self.consensus_energy + mean_energy_diff/scale)
            rv = math.log10(consensus_occupancy) - math.log10(mean_occ) - 6
            return rv
        res = bisect(f, 1e-1, 1e6)
        self.mean_energy = self.consensus_energy + mean_energy_diff/res
        self.motif_data = self.motif_data/res
        
        # change the units
        self.consensus_energy *= (R*T)
        self.mean_energy *= (R*T)
        self.motif_data *= (R*T)

        #print >> sys.stderr, self.factor
        #print >> sys.stderr, "Cons Energy:", self.consensus_energy
        #print >> sys.stderr, "Cons Occ:", logistic(self.consensus_energy/(R*T))
        #print >> sys.stderr, "Mean Energy:", self.mean_energy
        #print >> sys.stderr, "Mean Occ:", logistic(self.mean_energy/(R*T))
        #print >> sys.stderr, self.motif_data
        return

def build_wig(fasta, motif, region):
    chrm, start, stop, summit = region
    
    output = []

    print >> sys.stderr, "Processing %s:%i-%i\t(%i/%i)" % (
        chrm, start, stop, regions.qsize(), n_regions)
    seq = fasta.fetch(chrm, start, stop)
    max_score = -1e9
    best_pos = -1
    lines = []
    for pos, score in score_seq(seq, motif):
        score = logistic(score)
        if score > max_score:
            max_score = score
            best_pos = pos
        output.append( "%s\t%i\t%i\t%.2f\n" % (
            chrm, start + pos, start + pos + 1, score) )
    
    summit_line = "%i\t%i\t%.2f\t%.2f\n" % (
        best_pos, summit, 
        best_pos/float(stop-start), summit/float(stop-start))

    return output, summit_line

def build_wig_worker(fasta_fname, regions, motif):
    fasta = FastaFile(fasta_fname)
    n_regions = regions.qsize()
    output = []
    output_summits = []
    while not regions.empty():
        try: region = regions.get(timeout=0.1)
        except Queue.Empty: return
        region_output, summit = build_wig(fasta, motif, region)
        output.extend( region_output )
        output_summits.append(summit)
    
    ofp.write("".join(output))
    of_summits.write("".join(output))
    
    return

def build_wiggles_for_all_peaks(fasta_fname, proc_queue, binding_model):
    pids = []
    for i in xrange(24):
        pid = os.fork()
        if pid == 0:
            build_wig_worker()
            os._exit(0)
        else:
            pids.append(pid)
    for pid in pids:
        os.wait(pid, 0)

    return

def iter_motifs(fp):
    fp.seek(0)
    raw_motifs = fp.read().split(">")
    motifs = defaultdict(list)
    for motif_str in raw_motifs:
        #yield motif_str.split("\n")[0]
        if len(motif_str) == 0: continue
        yield Motif(motif_str)
    return 

def main():
    # missing 'TBP', 'TAF1', 'BCL11A'
    my_motifs = set(['CTCF', 'POU2F2', 'BATF', 'IRF4', 'REST', 'SPI1',
                     'MYC', 'NFKB', 'PAX5', 'TATA', 'TCF12', 'YY1'])
    print sorted(my_motifs)
    obs_factors = set()
    grpd_motifs = defaultdict(list)
    with open(sys.argv[1]) as fp:
        for motif in iter_motifs(fp):
            obs_factors.add(motif.factor)
            if motif.factor.upper() not in my_motifs:
                continue
            grpd_motifs[motif.factor].append(motif)

    #
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

        print factor, len([m.name for m in motifs])

    # choose a motif randomly
    for factor, motifs in sorted(grpd_motifs.items()):
        motif = motifs[0]
        with open("%s.motif.txt" % motif.name, "w") as ofp:
            ofp.write(">" + "\n".join(motif.lines) + "\n")

if __name__ == '__main__':
    main()
