import sys
import math
from collections import defaultdict, namedtuple

import numpy as np
from scipy.optimize import brute, bisect
from scipy.signal import fftconvolve, convolve
from scipy.stats.mstats import mquantiles

import h5py

from pysam import FastaFile

from sequence import one_hot_encode_sequences

from DB import load_genome_metadata

T = 300
R = 1.987e-3 # in kCal/mol*K

base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RC_base_map = {'A': 3, 'C': 2, 'G': 1, 'T': 0}

def logit(x):
    return math.log(x) - math.log(1-x)

def logistic(x):
    try: e_x = math.exp(-x)
    except: e_x = np.exp(-x)
    return 1/(1+e_x)

PwmModel = namedtuple('PwmModel', [
    'tf_id', 'motif_id', 'tf_name', 'tf_species', 'pwm']) 
SelexModelData = namedtuple('SelexModel', [
    'tf_id', 'motif_id', 'tf_name', 'tf_species', 
    'consensus_energy', 'ddg_array']) 
class SelexModel(SelexModelData):
    def __len__(self):
        return self.ddg_array.shape[0]

def load_pwms_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    import psycopg2
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
    SELECT tf_id, motif_id, tf_name, tf_species, pwm 
      FROM related_motifs_mv NATURAL JOIN pwms 
     WHERE tf_species in ('Mus_musculus', 'Homo_sapiens') 
       AND rank = 1 
    """

    if tf_names == None and tf_ids == None and motif_ids == None:
        cur.execute(query)
    elif tf_names != None and tf_ids == None and motif_ids == None:
        query += " AND tf_name in %s"
        cur.execute(query, [tuple(tf_names),])
    elif tf_ids != None and motif_ids == None and tf_names == None:
        query += " AND tf_id in %s"
        cur.execute(query, [tuple(tf_ids),])
    elif motif_ids != None and tf_ids == None and tf_names == None:
        query += " AND motif_id in %s"
        cur.execute(query, [tuple(motif_ids),])
    else:
        raise ValueError, "only one of tf_ids, tf_names, and motif_ids can can be set."
    
    motifs = []
    for data in cur.fetchall():
        data = list(data)
        data[-1] = np.log2(np.clip(1 - np.array(data[-1]), 1e-4, 1-1e-4))
        motifs.append( PwmModel(*data) )

    return motifs

def load_selex_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    import psycopg2
    conn = psycopg2.connect("host=mitra dbname=cisbp user=nboley")
    cur = conn.cursor()    
    query = """
     SELECT tf_id,
        format('SELEX_%%s', selex_motif_id) AS motif_id,
        tf_name,
        tf_species,
        consensus_energy,
        ddg_array
       FROM best_selex_models
    """

    #query = """
    #SELECT tf_id, selex_motif_id, tf_name, tf_species, consensus_energy, ddg_array 
    #  FROM best_selex_models
    #"""
    if tf_names == None and tf_ids == None and motif_ids == None:
        cur.execute(query, [])
    elif tf_names != None and tf_ids == None and motif_ids == None:
        query += " WHERE tf_name in %s"
        cur.execute(query, [tuple(tf_names),])
    elif tf_ids != None and motif_ids == None and tf_names == None:
        query += " WHERE tf_id in %s"
        cur.execute(query, [tuple(tf_ids),])
    elif motif_ids != None and tf_ids == None and tf_names == None:
        query += " WHERE selex_models.key in %s"
        cur.execute(query, [tuple(motif_ids),])
    else:
        raise ValueError, "only one of tf_ids, tf_names, and motif_ids can can be set."
    motifs = []
    for record in cur.fetchall():
        data = list(record)
        data[-1] = np.array(data[-1])
        data[-1][0,:] += record[4]
        motifs.append( SelexModel(*data) )

    if len(motifs) == 0:
        raise ValueError, "No motifs found (tf_ids: %s, tf_names: %s, motif_ids: %s)" % (
            tf_ids, tf_names, motif_ids)
    return motifs

def load_binding_models_from_db(tf_names=None, tf_ids=None, motif_ids=None):
    selex_motifs = load_selex_models_from_db(tf_names, tf_ids, motif_ids)
    cisb_motifs = load_pwms_from_db(tf_names, tf_ids, motif_ids)
    # Get one motif for each and prefer SELEX
    selex_tf_ids = set(m.tf_id for m in selex_motifs)
    return selex_motifs+[
        el for el in cisb_motifs if el.tf_id not in selex_tf_ids]


def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))

def _score_coded_seqs_binding_sites(coded_seqs, score_array):
    n_seqs = coded_seqs.shape[0]
    seq_len = coded_seqs.shape[1]
    n_bind_sites = seq_len-score_array.shape[0]+1

    # fft convolve is faster, but uses a lot more memory, so we 
    # dont use it for large regions
    convolve_fn = fftconvolve if seq_len < 1000000 else convolve
    
    rv = np.zeros((n_seqs, n_bind_sites), dtype='float32')
    for i, coded_seq in enumerate(coded_seqs):
        fwd_scores = convolve_fn(
            coded_seq, np.fliplr(np.flipud(score_array)),
            mode='valid')
        rc_scores = convolve_fn(coded_seq, score_array, mode='valid')
        rv[i,:] = np.hstack((fwd_scores, rc_scores)).max(1)
    
    return rv

def score_regions(regions, genome, motifs):
    seqs = [genome.fetch(region[0], region[1], region[2]) for region in regions]
    coded_seqs = one_hot_encode_sequences(seqs)
    for motif in motifs:
        if isinstance(motif, PwmModel):
            score_array = motif.pwm
        elif isinstance(motif, SelexModel):
            score_array = motif.ddg_array
        yield _score_coded_seqs_binding_sites(coded_seqs, score_array)
    return

def build_model_scores_hdf5_file(annotation_id, tf_id):
    genome_fname = load_genome_metadata(annotation_id).filename
    genome = FastaFile(genome_fname)
    models = load_binding_models_from_db(tf_ids=[tf_id,])
    assert len(models) == 1
    model = models[0]
    
    # open a file to write the scores to
    fname = "/srv/scratch/nboley/projects/hd5_scoring/binding_site_scores.ANNOTATIONID%i.MOTIFID%s.hdf5" % (
        annotation_id, model.motif_id)
    
    # check to see if the file exists and, if it does, raise an error
    try: 
        with open(fname): pass
    except IOError: 
        pass
    else: 
        raise IOError, "File '%s' already exists" % fname
    
    with h5py.File(fname, "w") as ofp:
        for i, (contig, contig_len) in enumerate(
                zip(genome.references, genome.lengths)):
            print >> sys.stderr, "Processing %i/%i" % (
                i+1, len(genome.references))
            scores = next(score_regions(
                [(contig, 0, contig_len),], genome, [model,]))
            dset = ofp.create_dataset(contig, scores.shape, dtype='f')
            dset[:] = scores

def score_regions_from_hdf5(regions, annotation_id, model):
    fname = "/srv/scratch/nboley/projects/hd5_scoring/binding_site_scores.ANNOTATIONID%i.MOTIFID%s.hdf5" % (annotation_id, model.motif_id)

    # if there are a lot of regins, it will be faster to cache the hdf5 file in memory
    if len(regions) > 20000:
        print "Pre-cacheing hdf5 file '%s'" % fname
        driver = 'core'
    else:
        driver = None

    with h5py.File(fname, 'r', driver=driver) as fp:
        for contig, start, stop in regions:
            yield ( (contig, start, stop), 
                    fp[contig][0, start:stop-len(model)+1] )
    return

def aggregate_region_scores(
        scores, quantile_probs = [0.99, 0.95, 0.90, 0.75, 0.50]):
    rv = [scores.mean()/len(scores), scores.max()]
    rv.extend(mquantiles(scores, prob=quantile_probs))
    return rv

def load_energy_data(fname):
    def load_energy(mo_text):
        lines = mo_text.strip().split("\n")
        motif_name, consensus_energy = lines[0].split()
        assert motif_name.endswith('.ENERGY')
        consensus_energy = float(consensus_energy)
        ddg_array = np.zeros((len(lines)-1,4))
        for pos, line in enumerate(lines[1:]):
            energies = np.array([float(x) for x in line.strip().split()[1:]])
            ddg_array[pos,:] = energies
        return consensus_energy, ddg_array
    
    with open(fname) as fp:
        models = fp.read().strip().split(">")
        # make sure there's a leading >
        assert models[0] == ''
        models = models[1:]
        assert len(models) == 2
        pwm_model_text = next(mo for mo in models if ".PWM" in mo.split("\n")[0])
        motif = load_motif_from_text(pwm_model_text)
        models.remove(pwm_model_text)
        assert len(models) == 1
        energy_mo_text = models[0]
        consensus_energy, ddg_array = load_energy(energy_mo_text)
        motif.update_energy_array(ddg_array, consensus_energy)
    
    return motif

def estimate_unbnd_conc_in_region(
        motif, score_cov, atacseq_cov, chipseq_rd_cov,
        frag_len, max_chemical_affinity_change):
    # trim the read coverage to account for the motif length
    trimmed_atacseq_cov = atacseq_cov[len(motif)+1:]
    chipseq_rd_cov = chipseq_rd_cov[len(motif)+1:]

    # normalzie the atacseq read coverage
    atacseq_weights = trimmed_atacseq_cov/trimmed_atacseq_cov.max()
    
    # build the smoothing window
    sm_window = np.ones(frag_len, dtype=float)/frag_len
    sm_window = np.bartlett(2*frag_len)
    sm_window = sm_window/sm_window.sum()

    def build_occ(log_tf_conc):
        raw_occ = logistic(log_tf_conc + score_cov/(R*T))
        occ = raw_occ*atacseq_weights
        smoothed_occ = np.convolve(sm_window, occ/occ.sum(), mode='same')

        return raw_occ, occ, smoothed_occ

    def calc_lhd(log_tf_conc):
        raw_occ, occ, smoothed_occ = build_occ(-log_tf_conc)
        #diff = (100*smoothed_occ - 100*rd_cov/rd_cov.sum())**2
        lhd = -(np.log(smoothed_occ + 1e-12)*chipseq_rd_cov).sum()
        #print log_tf_conc, diff.sum()
        return lhd

    res = brute(calc_lhd, ranges=(
        slice(0, max_chemical_affinity_change, 1.0),))[0]
    log_tf_conc = max(0, min(max_chemical_affinity_change, res))
                      
    return -log_tf_conc

class ReducedDeltaDeltaGArray(np.ndarray):
    def calc_base_contributions(self):
        base_contribs = np.zeros((self.motif_len, 4))
        base_contribs[:,1:4] = self.base_portion.T
        return base_contribs

    def calc_normalized_base_conts(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        ref_energy += base_contribs.min(1).sum()
        for i, min_energy in enumerate(base_contribs.min(1)):
            base_contribs[i,:] -= min_energy
        return ref_energy, base_contribs
    
    def calc_min_energy(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        return ref_energy + base_contribs.min(1).sum()

    def calc_max_energy(self, ref_energy):
        base_contribs = self.calc_base_contributions()
        return ref_energy + base_contribs.max(1).sum()

    def reverse_complement(self):
        rc_array = np.zeros(self.shape, dtype=self.dtype)
        ts_cont = float(self[2,:].sum())
        rc_array[(0,1),:] = self[(1,0),:]
        rc_array[:,:3] -= self[2,:3]
        return ts_cont, rc_array.view(DeltaDeltaGArray)[:,::-1]

    @property
    def base_portion(self):
        return self[:3,:]
    
    @property
    def shape_portion(self):
        assert self.shape[0] == 9
        return self[3:,:]

    @property
    def mean_energy(self):
        return self.sum()/self.shape[0]
    
    @property
    def motif_len(self):
        return self.shape[1]

    def consensus_seq(self):
        base_contribs = self.calc_base_contributions()
        return "".join( 'ACGT'[x] for x in np.argmin(base_contribs, axis=1) )

    def summary_str(self, ref_energy):
        rv = []
        rv.append(str(self.consensus_seq()))
        rv.append("Ref: %s" % ref_energy)
        rv.append(
            "Mean: %s" % (ref_energy + self.mean_energy))
        rv.append(
            "Min: %s" % self.calc_min_energy(ref_energy))
        rv.append("".join("{:>10}".format(x) for x in [
            'A', 'C', 'G', 'T', 'ProT', 'MGW', 'LHelT', 'RHelT', 'LRoll', 'RRoll']))
        for base_contribs in self.T.tolist():
            rv.append( 
                "".join(["      0.00",] + [
                    "{:10.2f}".format(x) for x in base_contribs]) 
            )
        return "\n".join(rv)
        pass

class Motif():
    def __len__(self):
        return self.length

    def iter_pwm_score(self, seq):
        seq = seq.upper()
        for offset in xrange(len(seq) - len(self)+1):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = 0.0
            RC_score = 0.0
            if 'N' in subseq: 
                yield offset + len(self)/2, False, 0.25*len(self)
                continue
            for i, base in enumerate(subseq):
                score += self.pwm[i][base_map[base]]
                RC_score += self.pwm[len(self)-i-1][RC_base_map[base]]
            RC = True if RC_score > score else False 
            yield offset, RC, max(score, RC_score)

    def iter_seq_score(self, seq):
        for offset in xrange(len(seq) - len(self)+1):
            subseq = seq[offset:offset+len(self)]
            assert len(self) == len(subseq)
            score = self.consensus_energy
            RC_score = self.consensus_energy
            if 'N' in subseq:
                yield offset, False, self.mean_energy
                continue
            for i, base in enumerate(subseq):
                if isinstance(subseq, str): base = base_map[base]
                score += self.motif_data[i][base]
                RC_score += self.motif_data[len(self)-i-1][3-base]
                #score += self.motif_data[i][base_map[base]]
                #RC_score += self.motif_data[len(self)-i-1][RC_base_map[base]]
            assert self.consensus_energy-1e-6 <= score <= self.max_energy+1e-6
            assert self.consensus_energy-1e-6 <= RC_score <= self.max_energy+1e-6
            RC = True if RC_score < score else False 
            yield offset, RC, min(score, RC_score)
            #yield offset, False, score

    def score_seq(self, seq):
        try: assert len(seq) >= len(self)
        except: 
            print seq
            raise
        return min(x[2] for x in self.iter_seq_score(seq))
    
    def est_occ(self, unbnd_tf_conc, seq):
        score = self.score_seq(seq)
        return logistic((unbnd_tf_conc - score)/(R*T))
    
    def build_occupancy_weights(self):
        for i, weights in enumerate(self.pwm):
            row = np.array([-logit(1e-3/2 + (1-1e-3)*x) 
                               for x in weights])
            min_val = row.min()
            self.consensus_energy += min_val
            row -= min_val
            self.motif_data[i, :] = row

        self.mean_energy = -2/(R*T)
        self.consensus_energy = (-2 + -1.5*len(self.pwm))/(R*T)        
        mean_energy_diff = sum(row.sum()/4 for row in self.motif_data)

        # mean_energy = self.consensus_energy + mean_energy_diff/scale
        # scale =  R*T*(self.consensus_energy + mean_energy_diff)/mean_energy
        scale = mean_energy_diff/(self.mean_energy - self.consensus_energy)
        self.motif_data /= scale
        
        # change the units
        self.consensus_energy *= (R*T)
        self.mean_energy *= (R*T)
        self.motif_data *= (R*T)

        assert self.min_energy == self.consensus_energy
        #print "Conc:", self.consensus_energy, logistic(-self.consensus_energy/(R*T))
        #print "Mean:", self.mean_energy, logistic(-self.mean_energy/(R*T))
        #print self.motif_data
        #assert False

    @property
    def min_energy(self):
        return self.consensus_energy + sum(x.min() for x in self.motif_data)

    @property
    def max_energy(self):
        return self.consensus_energy + self.motif_data.max(1).sum()

    def build_ddg_array(self, include_shape=False):
        ref_energy = self.consensus_energy
        energies = np.zeros(
            (3 + (6 if include_shape else 0), len(self)),
            dtype='float32')
        for i, base_energies in enumerate(self.motif_data):
            for j, base_energy in enumerate(base_energies[1:]):
                energies[j, i] = base_energy - base_energies[0]
            ref_energy += base_energies[0]
        return ref_energy, energies.view(DeltaDeltaGArray)

    def update_energy_array(self, ddg_array, ref_energy):
        assert self.motif_data.shape == ddg_array.shape
        self.motif_data = ddg_array.copy()
        self.consensus_energy = ref_energy
        # normalize so that the consensus base is zero at each position 
        for base_pos, base_data in enumerate(self.motif_data):
            min_energy = base_data.min()
            self.motif_data[base_pos,:] -= min_energy
            self.consensus_energy -= min_energy
        # update the mean energy
        self.mean_energy = self.consensus_energy + self.motif_data.mean()
        return
    
    def __init__(self, name, factor, pwm):
        self.name = name
        self.factor = factor
        
        self.lines = None
        self.meta_data_line = None

        print "hjere", pwm
        self.length = len(pwm)

        self.consensus_energy = 0.0
        self.motif_data = np.zeros((self.length, 4), dtype='float32')
                
        self.pwm = np.array(pwm, dtype='float32')
        
        self.build_occupancy_weights()
        return


def load_motif_from_text(text):
    # load the motif data
    lines = text.strip().split("\n")
    if lines[0][0] == '>': lines[0] = lines[0][1:]
    name = lines[0].split()[0]
    factor = name.split("_")[0]
    motif_length = len(lines)-1

    pwm = np.zeros((motif_length, 4), dtype=float)

    for i, line in enumerate(lines[1:]):
        pwm_row = np.array([
            float(x) for x in line.split()[1:]])
        pwm[i, :] = pwm_row

    motif = Motif(name, factor, pwm)
    motif.lines = lines
    motif.meta_data_line = lines[0]

    return motif

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
        yield load_motif_from_text(motif_str)
    return 

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
    
    return grpd_motifs
