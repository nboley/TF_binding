from collections import defaultdict

import random
import math

import numpy as np

from scipy.optimize import brentq
from scipy.signal import convolve

from pyTFbindtools.motif_tools import R, T
import pyTFbindtools.selex

PARTITION_FN_SAMPLE_SIZE = 1000

random_seqs = {}

def log_lhd_factory():
    import theano
    import theano.tensor as TT
    from theano.tensor.signal.conv import conv2d as theano_conv2d

    # ignore theano warnings
    import warnings
    warnings.simplefilter("ignore")

    sym_occ = TT.vector()
    sym_part_fn = TT.vector()
    sym_u = TT.scalar('u')
    sym_cons_dg = TT.scalar('cons_dg')
    sym_chem_pot = TT.scalar('chem_pot')
    sym_dna_conc = TT.scalar()

    sym_seqs = TT.tensor3(name='seqs')
    sym_ddg = TT.matrix(name='ddg')
    calc_energy_fn = theano.function(
        [sym_seqs, sym_ddg], theano_conv2d(sym_seqs, sym_ddg).min(2)[:,0])

    sym_e = TT.vector()

    calc_occ = theano.function([sym_chem_pot, sym_e], 
        1 / (1 + TT.exp((-sym_chem_pot+sym_e)/(R*T)))
    )

    calc_chem_pot_sum_term = theano.function(
        [sym_part_fn, sym_e, sym_u, sym_dna_conc], 
        (sym_part_fn*sym_dna_conc/(1+TT.exp((sym_e-sym_u)/(R*T)))).sum()
    ) 

    # calculate the sum of log occupancies for a particular round, given a 
    # set of energies
    calc_rnd_lhd_num = theano.function([sym_chem_pot, sym_cons_dg, sym_e], -(
        TT.log(1.0 + TT.exp(
            (-sym_chem_pot + sym_cons_dg + sym_e)/(R*T))).sum())
    )

    #@profile
    def calc_lhd_numerators(
            seq_ddgs, chem_affinities, ref_energy):
        # the occupancies of each rnd are a function of the chemical affinity of
        # the round in which there were sequenced and each previous round. We 
        # loop through each sequenced round, and calculate the numerator of the log 
        # lhd 
        numerators = {}
        for sequencing_rnd, seq_ddgs in seq_ddgs.iteritems():
            numerator = 0.0
            for rnd in xrange(1,sequencing_rnd+1):
                #numerator += np.log(
                #    logistic(-(-chem_affinities[rnd]+ref_energy+seq_ddgs)/(R*T))).sum()
                numerator += calc_rnd_lhd_num(
                    chem_affinities[rnd], ref_energy, seq_ddgs)
            numerators[sequencing_rnd] = numerator

        return numerators

    #@profile
    def est_chem_potential(
            partition_fn, energies, dna_conc, prot_conc ):
        """Estimate chemical affinity for round 1.

        [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
        exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
        """
        #@profile
        def f(u):
            sum_term = calc_chem_pot_sum_term(
                partition_fn, energies, u, dna_conc)
            return prot_conc - math.exp(u) - sum_term
        min_u = -1000
        max_u = 100 + math.log(prot_conc)
        rv = brentq(f, min_u, max_u, xtol=1e-4)
        return rv

    #@profile
    def calc_lhd_denominators_and_chem_pots(
            max_rnd_num,
            ref_energy, 
            ddg_array, 
            seq_len,
            dna_conc,
            prot_conc):
        # now calculate the denominator (the normalizing factor for each round)
        # calculate the expected bin counts in each energy level for round 0
        energies = ref_energy + calc_energy_fn(
            random_seqs[seq_len], ddg_array)
        expected_cnts = (4**seq_len)/float(len(random_seqs[seq_len]))
        curr_occupancies = expected_cnts*np.ones(
            len(energies), dtype=theano.config.floatX)

        denominators = {}
        chem_pots = {}
        for rnd in xrange(1,max_rnd_num+1):
            chem_affinity = est_chem_potential(
                energies, 
                curr_occupancies/curr_occupancies.sum(),
                dna_conc, 
                prot_conc )
            chem_pots[rnd] = chem_affinity
            curr_occupancies *= calc_occ(chem_affinity, energies)
            occs_sum = curr_occupancies.sum()
            if occs_sum > 0:
                denominators[rnd] = np.log(occs_sum)
            elif len(denominators) > 1:
                denominators[rnd] = (
                    denominators[-1] - (denominators[-1] - denominators[-2]))
            elif len(denominators) > 0:
                denominators[rnd] = denominators[-1]
            else:
                raise ValueError, "Minimum precision exceeded"
        return chem_pots, denominators

    #@profile
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     coded_seqs,
                     dna_conc,
                     prot_conc):
        seq_len = coded_seqs.values()[0].shape[2]
        if seq_len not in random_seqs:
            random_seqs[seq_len] = pyTFbindtools.selex.sample_random_coded_seqs(
                PARTITION_FN_SAMPLE_SIZE, seq_len)
        ref_energy = np.array(ref_energy).astype(theano.config.floatX)
        # score all of the sequences
        rnds_and_seq_ddgs = {}
        for rnd, rnd_coded_seqs in coded_seqs.iteritems():
            rnds_and_seq_ddgs[rnd] = calc_energy_fn(rnd_coded_seqs, ddg_array)

        # calcualte the denominators
        chem_affinities, denominators = calc_lhd_denominators_and_chem_pots(
            max(coded_seqs.keys()),
            ref_energy, 
            ddg_array,  
            seq_len,
            dna_conc,
            prot_conc)

        # calculate the numerators
        numerators = calc_lhd_numerators(
            rnds_and_seq_ddgs, chem_affinities, ref_energy)

        lhd = 0.0
        for rnd in rnds_and_seq_ddgs.keys():
            lhd += numerators[rnd] - len(rnds_and_seq_ddgs[rnd])*denominators[rnd]

        try:
            assert np.isfinite(lhd)
        except:
            print numerators
            print denominators
            raise
        return lhd

    return calc_log_lhd

def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))

def calc_binding_site_energies(coded_seqs, ddg_array):
    n_seqs = coded_seqs.shape[0]
    n_bind_sites = coded_seqs.shape[2]-ddg_array.shape[1]+1
    rv = np.zeros((n_seqs, n_bind_sites), dtype='float32')
    for i, coded_seq in enumerate(coded_seqs):
        rv[i,:] = convolve(coded_seq, ddg_array, mode='valid')
    return rv

#calc_log_lhd = None
calc_log_lhd = log_lhd_factory()
