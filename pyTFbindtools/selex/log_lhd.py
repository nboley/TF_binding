from collections import defaultdict

import random
import math

import numpy as np

from scipy.optimize import brentq
from scipy.signal import convolve

import theano
import theano.tensor as TT
from theano.tensor.signal.conv import conv2d as theano_conv2d
from theano.tensor.extra_ops import cumprod, diff as theano_diff
from theano.gradient import jacobian

from pyTFbindtools.motif_tools import R, T

def test_calc_affinities():
    """Place holder for some test code.

    """
    seq = 'GCGAATACC'
    coded_seq = code_seq(seq)
    RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    RC_seq = "".join(RC_map[x] for x in seq[::-1])
    coded_RC_seq = code_seq(RC_seq)

    print coded_seqs.shape
    print calc_affinities(coded_seqs, ddg_array )

def test_RC_equiv():
    """Make sure that the RC function works correctly.
    
    """
    import theano
    import theano.tensor as TT
    from theano.tensor.signal.conv import conv2d as theano_conv2d

    from pyTFbindtools.motif_tools import ReducedDeltaDeltaGArray
    import pyTFbindtools
    from pyTFbindtools import selex
    
    seqs = TT.tensor3(name='seqs', dtype=theano.config.floatX)
    ddg = TT.matrix(name='ddg', dtype=theano.config.floatX)
    fwd_bs_affinities = theano_conv2d(seqs, ddg[::-1,::-1])[:,0,:]
    #rc_bs_affinities = theano_conv2d(seqs, ddg)[:,0,:]
    #bs_affinities = TT.stack(
    #    (fwd_bs_affinities, rc_bs_affinities), axis=1).min(1)
    
    rc_ddg = (
        TT.concatenate((
            ddg[(1,0),:], 
            TT.zeros_like(
                ddg[(0,),:], dtype=theano.config.floatX)
        ), axis=0) 
        - ddg[2,:]
    )[:,::-1]
    rc_bs_affinities = (
        theano_conv2d(seqs, rc_ddg[::-1,::-1]) + ddg[2,:].sum()
    )[:,0,:]

    bs_affinities = TT.stack(
        (fwd_bs_affinities, rc_bs_affinities), axis=1).min(1)
    calc_bs_affinities = theano.function([seqs, ddg], bs_affinities )
    
    seq_affinities = bs_affinities.min(1)
    calc_affinities = theano.function([seqs, ddg], seq_affinities )
    
    seq = 'GCGAATACC'
    coded_seqs = code_seqs([seq,])
    RC_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    RC_seq = "".join(RC_map[x] for x in seq[::-1])
    coded_RC_seqs = code_seqs([RC_seq,])
    ddg_array = (
        np.array([[1,0.1,0.34], [0,0.34,0.1], [0,1,0], [0,0,1]], dtype='float32').T
    ).view(ReducedDeltaDeltaGArray)
    print calc_binding_site_energies(coded_seqs, ddg_array)
    print calc_bs_affinities(coded_seqs, ddg_array)
    print
    print calc_binding_site_energies(coded_RC_seqs, ddg_array)[:,::-1]
    print calc_bs_affinities(coded_RC_seqs, ddg_array)[:,::-1]
    print 
    return
    energy_diff, RC_ddg_array = ddg_array.reverse_complement()
    print energy_diff + calc_binding_site_energies(coded_seq, RC_ddg_array)
    print energy_diff + calc_binding_site_energies(coded_RC_seq, RC_ddg_array)[:,::-1]

    print -energy_diff + calc_binding_site_energies(coded_seq[None,(1,2,3),:], RC_ddg_array)
    return
    
def numerical_log_lhd_factory():
    import theano
    import theano.tensor as TT
    from theano.tensor.signal.conv import conv2d as theano_conv2d

    # ignore theano warnings
    import warnings
    warnings.simplefilter("ignore")

    ############################################################################
    ##
    ## Theano function to calculate reverse complment invariant affinities
    ##
    ############################################################################
    one_hot_seqs = TT.tensor3(name='one_hot_seqs', dtype=theano.config.floatX)
    ddg_base_cont = TT.matrix(name='ddg_base_cont', dtype=theano.config.floatX)
    fwd_bs_base_affinities = theano_conv2d(
        one_hot_seqs, ddg_base_cont[::-1,::-1])[:,0,:]
    rc_ddg_base_cont = (
        TT.concatenate((
            ddg_base_cont[(1,0),:], 
            TT.zeros_like(
                ddg_base_cont[(0,),:], dtype=theano.config.floatX)
        ), axis=0) 
        - ddg_base_cont[2,:]
    )[:,::-1]
    rc_bs_base_affinities = (
        theano_conv2d(one_hot_seqs, 
                      rc_ddg_base_cont[::-1,::-1]) + ddg_base_cont[2,:].sum()
    )[:,0,:]
    bs_affinities_without_shape = TT.stack(
        (fwd_bs_base_affinities, rc_bs_base_affinities), axis=1).min(1)

    ddg_shape_cont = TT.matrix(name='ddg_shape_cont', dtype=theano.config.floatX)
    fwd_shape_seqs = TT.tensor3(
        name='fwd_shape_seqs', dtype=theano.config.floatX)
    fwd_bs_shape_affinities = theano_conv2d(
        fwd_shape_seqs, ddg_shape_cont[::-1,::-1])[:,0,:]

    rc_shape_seqs = TT.tensor3(
        name='rc_shape_seqs', dtype=theano.config.floatX)
    rc_bs_shape_affinities = theano_conv2d(
        rc_shape_seqs, ddg_shape_cont)[:,0,:]
    

    fwd_bs_affinities = fwd_bs_base_affinities + fwd_bs_shape_affinities
    rc_bs_affinities = rc_bs_base_affinities + rc_bs_shape_affinities
    bs_affinities_with_shape = TT.stack(
        (fwd_bs_affinities, rc_bs_affinities), axis=1).min(1)
    
    #fwd_bs_affinities = fwd_bs_shape_affinities
    #rc_bs_affinities = rc_bs_shape_affinities
    
    theano_calc_affinities_with_shape = theano.function(
        [one_hot_seqs, fwd_shape_seqs, rc_shape_seqs, 
         ddg_base_cont, ddg_shape_cont], 
        bs_affinities_with_shape.min(1) )
    theano_calc_affinities_without_shape = theano.function(
        [one_hot_seqs, ddg_base_cont], 
        bs_affinities_without_shape.min(1) )

    def calc_affinities(seqs, ddg_array):
        if ddg_array.shape[0] == 9:
            return theano_calc_affinities_with_shape(
                seqs.one_hot_coded_seqs, 
                seqs.shape_coded_fwd_seqs, 
                seqs.shape_coded_RC_seqs,             
                ddg_array.base_portion,
                ddg_array.shape_portion)
        else:
            return theano_calc_affinities_without_shape(
                seqs.one_hot_coded_seqs, 
                ddg_array.base_portion)

    ############################################################################
    #
    # calculate binding occupancies for a set of energies and a single chemical
    # affinity
    #
    ############################################################################
    sym_e = TT.vector(dtype=theano.config.floatX)
    sym_chem_pot = TT.scalar('chem_pot', dtype=theano.config.floatX)
    calc_occ = theano.function([sym_chem_pot, sym_e], 
        1 / (1 + TT.exp((-sym_chem_pot+sym_e)/(R*T)))
    )


    ############################################################################
    #
    # XXX
    #
    ############################################################################
    sym_part_fn = TT.vector(dtype=theano.config.floatX)
    sym_u = TT.scalar('u', dtype=theano.config.floatX)
    sym_dna_conc = TT.scalar(dtype=theano.config.floatX)
    calc_chem_pot_sum_term = theano.function(
        [sym_part_fn, sym_e, sym_u, sym_dna_conc], 
        (sym_part_fn*sym_dna_conc/(1+TT.exp((sym_e-sym_u)/(R*T)))).sum()
    ) 

    ############################################################################
    #
    # calculate the sum of log occupancies for a particular round, given a 
    # set of energies
    #
    ############################################################################
    sym_cons_dg = TT.scalar('cons_dg', dtype=theano.config.floatX)
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
            background_seqs,
            max_rnd_num,
            ref_energy, 
            ddg_array, 
            seq_len,
            dna_conc,
            prot_conc):
        # now calculate the denominator (the normalizing factor for each round)
        # calculate the expected bin counts in each energy level for round 0
        energies = ref_energy + calc_affinities(background_seqs, ddg_array)
        expected_cnts = (4**seq_len)/float(background_seqs.n_seqs)
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
                    denominators[rnd-1] - 
                    (denominators[rnd-1] - denominators[rnd-2])
                )
            elif len(denominators) > 0:
                denominators[rnd] = denominators[rnd-1]
            else:
                raise ValueError, "Minimum precision exceeded"
        return chem_pots, denominators

    #@profile
    def calc_log_lhd(ref_energy, 
                     ddg_array, 
                     coded_seqs,
                     dna_conc,
                     prot_conc):
        ref_energy = np.array(ref_energy).astype(theano.config.floatX)
        # score all of the sequences
        rnds_and_seq_ddgs = {}
        for rnd, rnd_coded_seqs in coded_seqs.rnd_seqs.iteritems():
            rnds_and_seq_ddgs[rnd] = calc_affinities(
                rnd_coded_seqs, ddg_array)

        # calcualte the denominators
        chem_affinities, denominators = calc_lhd_denominators_and_chem_pots(
            coded_seqs.bg_seqs,
            coded_seqs.last_rnd_index,
            ref_energy, 
            ddg_array,
            coded_seqs.bg_seqs.seq_length,
            dna_conc,
            prot_conc)

        # calculate the numerators
        numerators = calc_lhd_numerators(
            rnds_and_seq_ddgs, chem_affinities, ref_energy)
        lhd = 0.0
        for rnd in rnds_and_seq_ddgs.keys():
            #print numerators[rnd], len(rnds_and_seq_ddgs[rnd])*denominators[rnd]
            lhd += numerators[rnd] - len(rnds_and_seq_ddgs[rnd])*denominators[rnd]

        try:
            assert np.isfinite(lhd)
        except:
            print numerators
            print denominators
            raise
        return lhd

    def calc_log_lhd_gradient(
            ref_energy, ddg_array, coded_seqs, dna_conc, prot_conc):
        pass

    return calc_log_lhd

def theano_calc_affinities(seqs, ref_energy, ddg):
    # ignore the A's in the forward direction by using seqs[:,:,1:]
    # flip the convolution along both axis to give the forward convolution
    # (because it's a true convolution, which means it's in the reverse
    #  direction...)
    fwd_bs_affinities = (
        ref_energy + theano_conv2d(seqs[:,:,1:], ddg[::-1,::-1])[:,:,0])
    # ignore the RC A's by ignoring the forward T's by using seqs[:,:,:3]
    # for the reverse complement affinities
    rc_bs_affinities = ref_energy + theano_conv2d(seqs[:,:,:3], ddg)[:,:,0]
    # stack the affinities, and then take the min at each bing site
    bs_affinities = TT.stack(fwd_bs_affinities, rc_bs_affinities).min(0)
    seq_affinities = bs_affinities.min(1)
    return seq_affinities

def theano_calc_occs(affinities, chem_pot):
    return 1 / (1 + TT.exp((-chem_pot+affinities)/(R*T)))

def theano_build_lhd_and_grad_fns(n_seqs):    
    dna_conc = TT.scalar(dtype=theano.config.floatX)
    prot_conc = TT.scalar(dtype=theano.config.floatX)
    
    n_rounds = len(n_seqs)

    rnd_seqs = [
        TT.tensor3(name='rnd_%iseqs' % (i+1), dtype=theano.config.floatX)
        for i in xrange(n_rounds) ]

    bg_seqs = TT.tensor3(name='bg_seqs', dtype=theano.config.floatX)
    seq_len = bg_seqs.shape[1]
    n_bg_seqs = bg_seqs.shape[0]
    
    ddg = TT.matrix(name='ddg', dtype=theano.config.floatX)
    ref_energy = TT.scalar(name='ref_energy', dtype=theano.config.floatX)

    chem_affinities = TT.vector(dtype=theano.config.floatX)

    # calculate the sequence affinities
    rnds_seq_affinities = [
        theano_calc_affinities(seqs, ref_energy, ddg) for seqs in rnd_seqs ]
    bg_seq_affinities = theano_calc_affinities(bg_seqs, ref_energy, ddg)

    # calculate the lhd numerators
    rnd_numerators = []
    for seq_rnd in xrange(1,n_rounds+1):
        rnd_numerators.append(TT.sum([
            -TT.log(1.0 + TT.exp(
                (-chem_affinities[i-1] + rnds_seq_affinities[seq_rnd-1])/(R*T))).sum()
            for i in xrange(1, seq_rnd+1) 
        ]))

    # calculate the lhd denominator
    expected_cnts = (4.0**seq_len)/n_bg_seqs
    bg_bindingsite_occs = TT.stack(
        [TT.ones_like(bg_seq_affinities)/bg_seq_affinities.shape[0],] + 
        [theano_calc_occs(bg_seq_affinities, chem_affinities[i-1])
         for i in xrange(1,n_rounds+1)])
    log_bnd_fracs = TT.log(TT.sum(cumprod(bg_bindingsite_occs, axis=0), axis=1))
    rnd_bnd_fracs = theano_diff(log_bnd_fracs)

    denominators = TT.log(expected_cnts) + log_bnd_fracs[1:]

    lhd = TT.sum(rnd_numerators - n_seqs*denominators)
    lhd_grad = jacobian(lhd, [ref_energy, ddg, chem_affinities])

    theano_calc_lhd = theano.function(
        [seqs for seqs in rnd_seqs] + [
            bg_seqs, ddg, ref_energy, chem_affinities, dna_conc, prot_conc],
        lhd )
    theano_calc_grad = theano.function(
        [seqs for seqs in rnd_seqs] + [
            bg_seqs, ddg, ref_energy, chem_affinities, dna_conc, prot_conc],
        lhd_grad)

    penalized_lhd = lhd - n_bg_seqs*TT.sum(abs(
        1 - chem_affinities/prot_conc - dna_conc*rnd_bnd_fracs/prot_conc))
    penalized_lhd_grad = jacobian(lhd, [ref_energy, ddg, chem_affinities])
    theano_calc_penalized_lhd = theano.function(
        [seqs for seqs in rnd_seqs] + [
            bg_seqs, ddg, ref_energy, chem_affinities, dna_conc, prot_conc],
        penalized_lhd )
    theano_calc_penalized_lhd_grad = theano.function(
        [seqs for seqs in rnd_seqs] + [
            bg_seqs, ddg, ref_energy, chem_affinities, dna_conc, prot_conc],
        penalized_lhd_grad)


    #theano_calc_bnd_fraction = theano.function(
    #    [seqs for seqs in rnd_seqs] + [
    #        bg_seqs, ddg, ref_energy, chem_affinities],
    #    rnd_bnd_fracs )


    return ( theano_calc_lhd, 
             theano_calc_grad, 
             theano_calc_penalized_lhd, 
             theano_calc_penalized_lhd_grad )

def theano_log_lhd_factory(initial_coded_seqs):
    rnds = initial_coded_seqs.rnd_seqs.keys()
    def get_num_seqs():
        n_seqs = []
        for rnd in xrange(1,max(rnds)+1):
            if rnd in initial_coded_seqs.rnd_seqs: 
                n_seqs.append(initial_coded_seqs.rnd_seqs[rnd].n_seqs)
            else:
                n_seqs.append(0)
        return np.array(n_seqs, dtype='float32')
    
    n_seqs = get_num_seqs()
    ( theano_calc_lhd, 
      theano_calc_grad, 
      theano_calc_penalized_lhd,
      theano_calc_penalized_lhd_grad
    ) = theano_build_lhd_and_grad_fns(n_seqs)
    
    chem_affinities = np.array([-11.0]*len(n_seqs), dtype='float32')
    
    def calc_lhd(ref_energy, ddg_array, coded_seqs, dna_conc, prot_conc):
        ref_energy = np.array([ref_energy,], dtype='float32')[0]
        coded_seqs_args = [
            coded_seqs.rnd_seqs[i].one_hot_coded_seqs for i in sorted(rnds)
        ] + [coded_seqs.bg_seqs.one_hot_coded_seqs,]
        args = coded_seqs_args + [
            ddg_array.T, ref_energy, chem_affinities, dna_conc, prot_conc]
        return theano_calc_lhd(*args)

    def calc_grad(ref_energy, ddg_array, coded_seqs, dna_conc, prot_conc):
        coded_seqs_args = [
            coded_seqs.rnd_seqs[i].one_hot_coded_seqs for i in sorted(rnds)
        ] + [coded_seqs.bg_seqs.one_hot_coded_seqs,]
        args = coded_seqs_args + [
            ddg_array, ref_energy, chem_affinities, dna_conc, prot_conc]
        return theano_calc_lhd_grad(*args)

    #def calc_bnd_frac(ddg_array, ref_energy, chem_affinities):
    #    args = coded_seqs_args + [ddg_array, ref_energy, chem_affinities]
    #    return theano_calc_bnd_fraction(*args)

    return calc_lhd, calc_grad

def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))

def calc_binding_site_energies(coded_seqs, ddg_array):
    n_bind_sites = coded_seqs.seq_length-ddg_array.shape[1]+1
    rv = np.zeros((coded_seqs.n_seqs, n_bind_sites), dtype='float32')
    for i, coded_seq in enumerate(coded_seqs.one_hot_coded_seqs):
        rv[i,:] = convolve(
            coded_seq, np.fliplr(np.flipud(ddg_array.base_portion)), 
            mode='valid')
    return rv

calc_log_lhd = None
#calc_log_lhd = numerical_log_lhd_factory()
#test_RC_equiv()
