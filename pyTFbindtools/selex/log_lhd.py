from collections import defaultdict

import random
import math

import numpy as np

from scipy.optimize import brentq
from scipy.signal import convolve

import theano
import theano.tensor as TT
from theano.tensor.signal.conv import conv2d as theano_conv2d
#from theano.tensor.nnet.conv import conv2d as theano_conv2d
#from theano.tensor.nnet.conv3d2d import conv3d as theano_conv2d

from theano.tensor.extra_ops import (
    cumsum as theano_cumsum, 
    diff as theano_diff,
    repeat as theano_repeat)
from theano.gradient import jacobian, hessian

from theano.compile.nanguardmode import NanGuardMode

from pyTFbindtools.motif_tools import R, T

def three_base_calc_affinities():
    # place to store code
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
            assert ddg_array.shape[2] == 4
            return theano_calc_affinities_without_shape(
                seqs.one_hot_coded_seqs, 
                ddg_array.base_portion)

def softmax(x, axis):
    scale_factor = x.max(axis=axis, keepdims=True)
    e_x = TT.exp(x - scale_factor)
    weights = e_x/e_x.sum(axis=axis, keepdims=True)
    return (x*weights).sum(axis)

def theano_calc_affinities(seqs, ref_energy, ddg, ddg_shape_cont):    
    # flip the convolution along both axis to give the forward convolution
    # (because it's a true convolution, which means it's in the reverse
    #  direction...)
    fwd_bs_base_affinities = (
        ref_energy + theano_conv2d(seqs[:,:,0:4], ddg[::-1,::-1])[:,:,0])
    # we don't flip the filter because the convolutions implicitly flip it
    # giving us the RC directly
    rc_bs_base_affinities = ref_energy + theano_conv2d(seqs[:,:,0:4], ddg)[:,:,0]

    # stack the affinities, and then take the min at each binding site
    if ddg_shape_cont is not None:
        fwd_bs_shape_affinities = theano_conv2d(
            seqs[:,:,4:10], ddg_shape_cont[::-1,::-1])[:,:,0]
        fwd_bs_affinities = fwd_bs_base_affinities + fwd_bs_shape_affinities

        rc_bs_shape_affinities = theano_conv2d(
            seqs[:,:,10:16], ddg_shape_cont)[::-1,:,0]
        rc_bs_affinities = rc_bs_base_affinities + rc_bs_shape_affinities
    else:
        fwd_bs_affinities = fwd_bs_base_affinities
        rc_bs_affinities = rc_bs_base_affinities

    bs_affinities = TT.stack(fwd_bs_affinities, rc_bs_affinities).min(0) 
    seq_affinities = -softmax(-bs_affinities, axis=1) # bs_affinities.min(1)

    return seq_affinities

def theano_calc_log_occs(affinities, chem_pot):
    inner = (-chem_pot+affinities)/(R*T)
    lower = TT.switch(inner>35, inner, 0)
    mid = TT.switch((inner >= -10)&(inner <= 35), 
                    TT.log(1.0 + TT.exp(inner)),
                    0 )
    upper = TT.switch(inner>35, inner, 0)
    return -(lower + mid + upper)

def theano_log_sum_log_occs(log_occs):
    # theano.printing.Print('scale_factor')
    scale_factor = (
        TT.max(log_occs, axis=1, keepdims=True))
    centered_log_occs = (log_occs - scale_factor)
    centered_rv = TT.log(TT.sum(TT.exp(centered_log_occs), axis=1))
    return centered_rv + scale_factor.flatten()

def relu(x):
    return (x + abs(x))/2

def theano_round_log_occs(seq_affinities, chem_affinities, rnd):
    rv = 0
    for i in xrange(rnd+1):
        rv += theano_calc_log_occs(
            seq_affinities, chem_affinities[i]
        ).sum()
    return rv

def theano_build_lhd_and_grad_fns(data):
    use_shape = data.have_shape_features
    chem_affinities = TT.vector(dtype=theano.config.floatX)

    bg_seqs = data.bg_seqs.fwd_and_rev_coded_seqs
    #bg_seqs = TT.tensor3(name='bg_seqs', dtype=theano.config.floatX)
    n_bg_seqs = bg_seqs.shape[0]

    seq_len = bg_seqs.shape[1]

    rnd_seqs = [seqs.fwd_and_rev_coded_seqs for seqs in data.rnd_seqs.values()]
    n_rounds = len(rnd_seqs)
    
    #rnd_seqs = [
    #    TT.tensor3(name='rnd_%iseqs' % (i+1), dtype=theano.config.floatX)
    #    for i in xrange(n_rounds) ]    
    n_seqs = TT.ones_like(chem_affinities)
    for rnd in xrange(n_rounds):
        n_seqs = TT.set_subtensor(n_seqs[rnd], rnd_seqs[rnd].shape[0])

    if use_shape:
        n_features_per_base = 10
    else:
        n_features_per_base = 4
    
    dna_conc = TT.scalar(dtype=theano.config.floatX)
    prot_conc = TT.scalar(dtype=theano.config.floatX)
    
    ddg_flat = TT.vector(name='ddg', dtype=theano.config.floatX)
    motif_len = ddg_flat.shape[0]/n_features_per_base
    ddg = ddg_flat.reshape((motif_len, n_features_per_base))
    ddg_base_portion = ddg[:,:4]
    ddg_shape_cont = ddg[:,4:]

    ref_energy = TT.scalar(name='ref_energy', dtype=theano.config.floatX)

    
    # calculate the sequence affinities
    rnds_seq_affinities = [
        theano_calc_affinities(
            seqs, ref_energy, ddg_base_portion, ddg_shape_cont) 
        for seqs in rnd_seqs ]
    bg_seq_affinities = theano_calc_affinities(
        bg_seqs, ref_energy, ddg_base_portion, ddg_shape_cont)

    # calculate the lhd numerators
    rnd_numerators = TT.zeros_like(chem_affinities)
    for seq_rnd in xrange(n_rounds):
        rnd_numerators = TT.inc_subtensor(
            rnd_numerators[seq_rnd],
            theano_round_log_occs(
                rnds_seq_affinities[seq_rnd], chem_affinities, seq_rnd)
        )
    
    # calculate the lhd denominator
    # calculate the log bound occupancies for each background sequence
    # for each chemical affinity
    # XXX - can we eliminate the list comprehension???
    bg_bindingsite_log_occs = TT.stack([
        theano_calc_log_occs(bg_seq_affinities, chem_affinities[i-1])
        for i in xrange(1,n_rounds+1)
    ])    
    expected_cnts = (4.0**seq_len)/n_bg_seqs
    denominators = theano_log_sum_log_occs(
        theano_cumsum(
            bg_bindingsite_log_occs, axis=0)
    ) + TT.log(expected_cnts)
    
    bg_bindingsite_log_occs = TT.stack([
        theano_calc_log_occs(bg_seq_affinities, chem_affinities[i-1])
        for i in xrange(1,n_rounds+1)
    ])
    bg_bindingsite_log_weights = TT.ones_like(bg_bindingsite_log_occs)
    for i in xrange(n_rounds):
        # if we are in round 0 initialize to sum to be uniform across seqs
        if i == 0:
            bg_bindingsite_log_weights = TT.set_subtensor(
                bg_bindingsite_log_weights[i,:],
                -TT.log(n_bg_seqs)
            )
        # otherwise, initialize to the previous round, and then normalize
        else:
            bg_bindingsite_log_weights = TT.set_subtensor(
                bg_bindingsite_log_weights[i,:],
                bg_bindingsite_log_weights[i-1,:]
            )
            # normalize so that the probabilities to sum to 1
            bg_bindingsite_log_weights = TT.inc_subtensor(
                bg_bindingsite_log_weights[i,:],
                -theano_log_sum_log_occs(bg_bindingsite_log_weights)[i]
            )

        # update with the current probabilities
        bg_bindingsite_log_weights = TT.inc_subtensor(
            bg_bindingsite_log_weights[i,:],
            bg_bindingsite_log_occs[i,:]
        )
    # sum up the bg log ocucpancies. Since the weights are initialzied to
    # sum to one, this is actually the mean occupancy of the background seqs
    bg_log_occs = (
        theano_log_sum_log_occs(bg_bindingsite_log_weights))

    print_numerators = theano.printing.Print('numerators')
    print_denominators = theano.printing.Print('denominators')
    # TT.log(expected_cnts) + 
    #lhd = (print_numerators(rnd_numerators) 
    #       - n_seqs*print_denominators(denominators)).sum()
    lhd = (rnd_numerators - n_seqs*denominators).sum()
    
    #print_scaled_denom = theano.printing.Print('scaled_denom')
    #print_numerators_str = theano.printing.Print('num_str')
    #lhd = TT.sum(print_numerators_str(rnd_numerators) - print_scaled_denom(1000*denominators))

    theano_calc_lhd, theano_calc_grad = None, None
    theano_calc_lhd = theano.function(
        [ddg_flat, ref_energy, chem_affinities, dna_conc, prot_conc],
        lhd )
    #theano_calc_grad = theano.function(
    #    [seqs for seqs in rnd_seqs] + [
    #        bg_seqs, ddg, ref_energy, chem_affinities, dna_conc, prot_conc],
    #    lhd_grad)

    theano_calc_log_unbnd_imbalance = None
    print_bnd_frac = theano.printing.Print('bnd_frac')
    log_unbnd_imbalance = (  
        TT.log(prot_conc) 
        - theano_log_sum_log_occs(
            TT.stack(
                chem_affinities,
                TT.log(dna_conc) + bg_log_occs
            ).transpose()
        )
    )

    theano_calc_log_unbnd_imbalance = theano.function(
        [ddg_flat, ref_energy, chem_affinities, dna_conc, prot_conc],
        log_unbnd_imbalance,
        allow_input_downcast=True)

    print_mean_energy = theano.printing.Print('mean_energy')
    #mean_energy = print_mean_energy(bg_seq_affinities.sum()/n_bg_seqs)
    mean_energy = ref_energy+ddg_base_portion.sum()/4
    penalized_lhd = ( 
        lhd 
        #- 100*TT.sum(log_unbnd_imbalance**2)
        - 100*TT.exp((-6-mean_energy)**2)
        - 100*TT.sum(TT.exp(-2+TT.max(ddg_base_portion, axis=1)
                            - TT.min(ddg_base_portion, axis=1)))
    )
    #penalized_lhd = lhd
    penalized_lhd_grad = jacobian(
        penalized_lhd, [ref_energy, ddg, chem_affinities])

    #penalized_lhd_hessian = hessian(
    #    penalized_lhd, wrt=ddg_flat)
    
    #theano_calc_penalized_lhd = None
    theano_calc_penalized_lhd = theano.function(
        [ddg_flat, ref_energy, chem_affinities, dna_conc, prot_conc],
        penalized_lhd,
        allow_input_downcast=True)
    theano_calc_penalized_lhd_grad = None
    theano_calc_penalized_lhd_grad = theano.function(
        [ddg_flat, ref_energy, chem_affinities, dna_conc, prot_conc],
        penalized_lhd_grad,
        allow_input_downcast=True
        #mode=NanGuardMode(nan_is_error=False, inf_is_error=False, big_is_error=False)
    )
    theano_calc_penalized_lhd_hessian = None
    #theano_calc_penalized_lhd_hessian = theano.function(
    #    [seqs for seqs in rnd_seqs] + [
    #        bg_seqs, ddg_flat, ref_energy, chem_affinities, dna_conc, prot_conc],
    #    penalized_lhd_hessian,
    #    allow_input_downcast=True
    #    #mode=NanGuardMode(nan_is_error=False, inf_is_error=False, big_is_error=False)
    #)
    
    #theano_calc_bnd_fraction = theano.function(
    #    [seqs for seqs in rnd_seqs] + [
    #        bg_seqs, ddg, ref_energy, chem_affinities],
    #    rnd_bnd_fracs )


    return ( theano_calc_lhd, 
             theano_calc_grad, 
             theano_calc_penalized_lhd, 
             theano_calc_penalized_lhd_grad,
             theano_calc_penalized_lhd_hessian,
             theano_calc_log_unbnd_imbalance)

def theano_log_lhd_factory(initial_coded_seqs):
    rnds = initial_coded_seqs.rnd_seqs.keys()
    ( theano_calc_lhd, 
      theano_calc_grad, 
      theano_calc_penalized_lhd,
      theano_calc_penalized_grad,
      theano_calc_penalized_hessian,
      theano_calc_log_unbnd_imbalance
    ) = theano_build_lhd_and_grad_fns(initial_coded_seqs)
    
    def build_args(ref_energy, ddg_array, chem_affinities,
                  coded_seqs, 
                  dna_conc, prot_conc):
        coded_seqs_args = []
        #for i in sorted(rnds):
        #    coded_seqs_args.append(coded_seqs.rnd_seqs[i].coded_seqs)
        #coded_seqs_args.append(coded_seqs.bg_seqs.coded_seqs)

        ref_energy = np.array([ref_energy,], dtype='float32')[0]
        args = coded_seqs_args + [
            ddg_array.ravel(), ref_energy, chem_affinities, dna_conc, prot_conc]
        return args

    def calc_lhd(ref_energy, ddg_array, chem_affinities, 
                 coded_seqs, 
                 dna_conc, prot_conc,
                 penalized=True):
        args = build_args(
            ref_energy, ddg_array, chem_affinities,
            coded_seqs, 
            dna_conc, prot_conc
        )
        if penalized:
            return theano_calc_penalized_lhd(*args)
        else:
            return theano_calc_lhd(*args)

    def calc_grad(ref_energy, ddg_array, chem_affinities,
                  coded_seqs, 
                  dna_conc, prot_conc):
        args = build_args(
            ref_energy, ddg_array, chem_affinities,
            coded_seqs, 
            dna_conc, prot_conc
        )
        return theano_calc_penalized_grad(*args)

    def calc_hessian(ref_energy, ddg_array, chem_affinities,
                     coded_seqs, 
                     dna_conc, prot_conc):
        args = build_args(
            ref_energy, ddg_array, chem_affinities,
            coded_seqs, 
            dna_conc, prot_conc
        )
        return theano_calc_penalized_hessian(*args)
    calc_hessian = None
    
    def calc_log_unbnd_imbalance(
            ref_energy, ddg_array, chem_affinities,
            coded_seqs, 
            dna_conc, prot_conc):
        return theano_calc_log_unbnd_imbalance(
            ddg_array.ravel(), ref_energy, chem_affinities,
            dna_conc, prot_conc)

    #def calc_bnd_frac(ddg_array, ref_energy, chem_affinities):
    #    args = coded_seqs_args + [ddg_array, ref_energy, chem_affinities]
    #    return theano_calc_bnd_fraction(*args)

    return calc_lhd, calc_grad, calc_hessian, calc_log_unbnd_imbalance

def calc_occ(chem_pot, energies):
    return 1. / (1. + np.exp((-chem_pot+energies)/(R*T)))

def calc_binding_site_energies(coded_seqs, ddg_array):
    n_bind_sites = coded_seqs.seq_length-ddg_array.motif_len+1
    rv = np.zeros((coded_seqs.n_seqs, n_bind_sites), dtype='float32')
    for i, coded_seq in enumerate(coded_seqs.one_hot_coded_seqs):
        rv[i,:] = convolve(
            coded_seq, np.fliplr(np.flipud(ddg_array.base_portion)), 
            mode='valid')
    return rv

calc_log_lhd = None
