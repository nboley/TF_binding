import sys

import math

import numpy as np 
np.random.seed(0)
import random
random.seed(0)

from scipy.optimize import brentq

import theano
import theano.tensor as TT
from theano.tensor.signal.conv import conv2d as theano_conv2d
from theano.tensor.extra_ops import cumprod, diff as theano_diff
from theano.gradient import jacobian

from pyTFbindtools.selex.log_lhd import calc_log_lhd
from pyTFbindtools.selex import code_seqs, SelexData, ReducedDeltaDeltaGArray
from pyDNAbinding.binding_model import FixedLengthDNASequences
from pyDNAbinding.sequence import sample_random_seqs
from pyDNAbinding.misc import load_fastq, optional_gzip_open, calc_occ
from pyDNAbinding.DB import load_binding_models_from_db

T = 300
R = 1.987e-3 # in kCal/mol*K

n_dna_seq = 7.5e-8/(1.079734e-21*119) # molecules  - g/( g/oligo * oligo/molecule)
jolma_dna_conc = n_dna_seq/(6.02e23*5.0e-5) # mol/L
n_water_mol = 55.55 #mol/L
jolma_prot_conc = jolma_dna_conc/25 # mol/L (should be 25)

total_mols = jolma_dna_conc + n_water_mol + jolma_prot_conc
DNA_molar_frac = jolma_dna_conc/total_mols
prot_molar_frac = jolma_prot_conc/total_mols

def old_code_seqs(rnds_and_seqs, bg_seqs):
    coded_rnds_and_seqs = dict(
        (rnd, code_seqs(seqs)) for rnd, seqs in rnds_and_seqs.iteritems())
    return SelexData( code_seqs(bg_seqs), coded_rnds_and_seqs )
                      

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

    # XXX really messy hack to append a 0  
    denominators = TT.log(expected_cnts) + log_bnd_fracs[1:]

    lhd = TT.sum(rnd_numerators - n_seqs*denominators)
    lhd_grad = jacobian(lhd, [ref_energy, ddg, chem_affinities])

    theano_calc_bnd_fraction = theano.function(
        [seqs for seqs in rnd_seqs] + [bg_seqs, ddg, ref_energy, chem_affinities],
        rnd_bnd_fracs )

    theano_calc_lhd = theano.function(
        [seqs for seqs in rnd_seqs] + [bg_seqs, ddg, ref_energy, chem_affinities],
        lhd )
    
    theano_calc_grad = theano.function(
        [seqs for seqs in rnd_seqs] + [bg_seqs, ddg, ref_energy, chem_affinities],
        lhd_grad)

    return theano_calc_lhd, theano_calc_grad, theano_calc_bnd_fraction

def get_num_seqs(coded_seqs):
    n_seqs = []
    for rnd in xrange(1,max(coded_seqs.keys())+1):
        if rnd in coded_seqs: 
            n_seqs.append(coded_seqs[rnd].shape[0])
        else:
            n_seqs.append(0)
    return np.array(n_seqs, dtype='float32')

def calc_lhd_factory(coded_seqs, coded_bg_seqs):
    n_seqs = get_num_seqs(coded_seqs)
    (theano_calc_lhd, theano_calc_grad, theano_calc_bnd_fraction
     ) = theano_build_lhd_and_grad_fns(n_seqs)
    
    coded_seqs_args = [
        coded_seqs[i] for i in sorted(coded_seqs.keys())
    ] + [coded_bg_seqs,]
    def calc_lhd(ddg_array, ref_energy, chem_affinities):
        args = coded_seqs_args + [ddg_array, ref_energy, chem_affinities]
        return theano_calc_lhd(*args)

    def calc_grad(ddg_array, ref_energy, chem_affinities):
        args = coded_seqs_args + [ddg_array, ref_energy, chem_affinities]
        return theano_calc_grad(*args)

    def calc_bnd_frac(ddg_array, ref_energy, chem_affinities):
        args = coded_seqs_args + [ddg_array, ref_energy, chem_affinities]
        return theano_calc_bnd_fraction(*args)

    return calc_lhd, calc_grad, calc_bnd_frac

def load_sequences(fnames, max_num_seqs_per_file=1e8):
    fnames = list(fnames)
    rnds_and_seqs = {}
    rnd_nums = [int(x.split("_")[-1].split(".")[0]) for x in fnames]
    rnds_and_fnames = dict(zip(rnd_nums, fnames))
    for rnd, fname in rnds_and_fnames.iteritems():
        with optional_gzip_open(fname) as fp:
            rnds_and_seqs[rnd] = load_fastq(fp, max_num_seqs_per_file)
    return rnds_and_seqs

def short_seq_test():
    seq_len = 5
    bg_seqs = sample_random_seqs(2, seq_len)
    rnd_seqs = {1: ['A'*5,'T'*5, 'ACCGT', 'CCGTT'], 2: ['A'*5]}
    coded_seqs = dict(
        (rnd, FixedLengthDNASequences(rnd_seqs[rnd]).one_hot_coded_seqs)
        for rnd in rnd_seqs.keys() )
    coded_bg_seqs = FixedLengthDNASequences(bg_seqs).one_hot_coded_seqs
    return coded_bg_seqs, coded_seqs

def main():
    with open(sys.argv[1]) as fp:
        bg_seqs = load_fastq(fp)
    coded_bg_seqs = FixedLengthDNASequences(bg_seqs).one_hot_coded_seqs

    rnd_seqs = load_sequences(sys.argv[2:])
    coded_seqs = dict(
        (rnd, FixedLengthDNASequences(rnd_seqs[rnd]).one_hot_coded_seqs)
        for rnd in rnd_seqs.keys() )

    calc_lhd, calc_grad, calc_bnd_frac = calc_lhd_factory(
        coded_seqs, coded_bg_seqs)

    chem_affinities = np.array([-8.1]*len(rnd_seqs), dtype='float32')    
    mo = load_binding_models_from_db(tf_names=['MAX',])[0]
    ref_energy, ddg_array = mo.build_all_As_affinity_and_ddg_array()
    ddg_array = ddg_array.T.astype('float32')

    affinities = -FixedLengthDNASequences(bg_seqs).score_binding_sites(
        mo, 'MAX').max(1)
    occs = calc_occ(chem_affinities[0], affinities)
    print affinities.min(), affinities.mean(), affinities.max()
    print occs.min(), occs.mean(), occs.max()
    print np.exp(calc_bnd_frac(ddg_array, ref_energy, chem_affinities))
    return
    print chem_affinities
    print ref_energy
    print ddg_array
    
    def est_chem_potential():
        """Estimate chemical affinity for round 1.

        [TF] - [TF]_0 - \sum{all seq}{ [s_i]_0[TF](1/{[TF]+exp(delta_g)}) = 0  
        exp{u} - [TF]_0 - \sum{i}{ 1/(1+exp(G_i)exp(-)
        """
        #@profile
        prot_conc = np.array([jolma_prot_conc]*len(chem_affinities))
        def f(u):
            chem_affinities[i] = u
            bnd_frac = calc_bnd_frac(ddg_array, ref_energy, chem_affinities)
            #print jolma_prot_conc
            #print chem_affinities, chem_affinities[i], math.exp(chem_affinities[i])
            #print bnd_frac, bnd_frac[i], math.exp(bnd_frac[i])
            #print
            rv = ( jolma_prot_conc 
                   - math.exp(chem_affinities[i]) 
                   - jolma_dna_conc*math.exp(bnd_frac[i]) )
            return rv
        
        for i in xrange(4):
            min_u = -1000
            max_u = 10 # + math.log(prot_conc)
            rv = brentq(f, min_u, max_u, xtol=1e-4)
            chem_affinities[i] = rv
        
        return chem_affinities
    
    #chem_affinities = est_chem_potential()
    for i in xrange(200):
        print i, chem_affinities, calc_lhd(ddg_array, ref_energy, chem_affinities)
        print calc_bnd_frac(ddg_array, ref_energy, chem_affinities)
        print ref_energy
        print ddg_array
        ref_energy_grad, ddg_grad, chem_affinity_grad = calc_grad(
            ddg_array, ref_energy, chem_affinities)
        #ref_energy += ref_energy_grad.clip(-1,1)
        chem_affinities = est_chem_potential()
        #chem_affinities += chem_affinity_grad
        ddg_array += ddg_grad #.clip(-1.0, 1.0)
        print
    return

    old_ddg_array = np.array(
        [[1,1,1], [1,1,1], [1,1,1], [1,1,1]], dtype='float32').T.view(
            ReducedDeltaDeltaGArray)
    print old_code_seqs(rnd_seqs, bg_seqs).bg_seqs
    print "old lhd", calc_log_lhd(
        -4, old_ddg_array, old_code_seqs(rnd_seqs, bg_seqs), 
        jolma_dna_conc, jolma_prot_conc)
    #print calc_log_lhd(-1, ddg_array, {1: coded_seqs}, jolma_dna_conc, jolma_prot_conc)

main()
