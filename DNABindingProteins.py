import os

import numpy

from grit.files.reads import Reads, MergedReads, determine_read_pair_params
from grit.frag_len import build_normal_density

class ChIPSeqReads(Reads):
    def __repr__(self):
        paired = 'paired' if self.reads_are_paired else 'unpaired'
        return "<ChIPSeqReads.%s.%i instance>" % (paired, self.frag_len)

    def build_unpaired_reads_fragment_coverage_array( 
            self, chrm, strand, start, stop, binding_site_size, 
            window_size=None ):
        if window_size == None:
            window_size = self.frag_len
        assert stop >= start
        full_region_len = stop - start + 1
        cvg = numpy.zeros(full_region_len)
        for rd, strand in self.iter_reads_and_strand(chrm, start, stop):
            if strand == '+': 
                rd_start = rd.pos # + binding_site_size
                rd_stop = rd.pos + window_size # + binding_site_size
            elif strand == '-': 
                rd_start = rd.aend - window_size
                rd_stop = rd.aend
            else:
                assert False
            cvg[max(0, rd_start-start):max(0, rd_stop-start)] += (
                1.0/(rd_stop-rd_start+1))

        return cvg

        
    def init(self, 
             reverse_read_strand=None,  reads_are_stranded=None,
             pairs_are_opp_strand=None, reads_are_paired=None,
             frag_len_dist=None):        
        assert self.is_indexed()

        "ChIPSeq.GM12878.CTCF.BSID-ENCBS195XMM.REPID-1_1.EXPID-ENCSR000DRZ.bam"
        data = os.path.basename(self.filename).split('.')
        self.biosample = data[1] 
        self.factor = data[2]
        self.bsid = data[3].split("-")[1]
        self.experiment_id = data[5].split("-")[1]
        self.repid = data[4].split("-")[1]

        self.id = "%s.%s.%s" % (self.factor, self.bsid, self.repid)
        
        reads_are_stranded = True
        
        if frag_len_dist == None:
            frag_len_dist = build_normal_density(
                fl_min=100, fl_max=200, mean=150, sd=25)
        self.frag_len_dist = frag_len_dist
        self.frag_len = int(frag_len_dist.mean_fragment_length())
        
        read_pair_params = determine_read_pair_params(self)
        
        # set whether the reads are paired or not
        if reads_are_paired in ('auto', None):
            if 'paired' in read_pair_params:
                reads_are_paired = True 
                assert 'unpaired' in read_pair_params
            else:
                reads_are_paired = False
        
        if pairs_are_opp_strand in ('auto', None):
            if reads_are_paired or ('same_strand' in read_pair_params):
                pairs_are_opp_strand = False
            else:
                pairs_are_opp_strand = True
        
        reverse_read_strand = None
        
        Reads.init(self, reads_are_paired, pairs_are_opp_strand, 
                         reads_are_stranded, reverse_read_strand )

        # we save these for fast reloads
        self._init_kwargs = {
            'reverse_read_strand': reverse_read_strand, 
            'reads_are_stranded': reads_are_stranded, 
            'pairs_are_opp_strand': pairs_are_opp_strand, 
            'reads_are_paired': reads_are_paired
        }
        
        return self

def get_chipseq_experiment(ENCODE_exp_ID):
    pass
