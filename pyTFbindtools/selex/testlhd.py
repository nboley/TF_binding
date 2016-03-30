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
from pyDNAbinding.binding_model import (
    FixedLengthDNASequences, est_chem_potential_from_affinities )
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

dna_conc = jolma_dna_conc
prot_conc = jolma_prot_conc

def old_code_seqs(rnds_and_seqs, bg_seqs):
    coded_rnds_and_seqs = dict(
        (rnd, code_seqs(seqs)) for rnd, seqs in rnds_and_seqs.iteritems())
    return SelexData( code_seqs(bg_seqs), coded_rnds_and_seqs )
                      

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

from log_lhd import theano_build_lhd_and_grad_fns

def main():
    with open(sys.argv[1]) as fp:
        bg_seqs = load_fastq(fp)
    coded_bg_seqs = FixedLengthDNASequences(bg_seqs).one_hot_coded_seqs

    rnd_seqs = load_sequences(sys.argv[2:])
    coded_seqs = dict(
        (rnd, FixedLengthDNASequences(rnd_seqs[rnd]).one_hot_coded_seqs)
        for rnd in rnd_seqs.keys() )

    _, _, log_lhd, log_lhd_grad = theano_build_lhd_and_grad_fns(
        max(rnd_seqs.keys()))
    print log_lhd
    
    mo = load_binding_models_from_db(tf_names=['MAX',])[0]
    ref_energy, ddg_array = 0.0, -mo.ddg_array
    chem_affinities = np.array([-21.0]*len(rnd_seqs), dtype='float32')
    args = [coded_seqs[i] for i in sorted(coded_seqs.keys())]
    args.extend( [coded_bg_seqs, 
                  ddg_array, ref_energy, chem_affinities, 
                  dna_conc, prot_conc] )
    print ddg_array.shape
    print coded_bg_seqs.shape
    print [x.shape for x in args[:7]]
    print log_lhd(*args)
    #print log_lhd_grad(*args)
    return
        
    for i in xrange(200):
        print i, calc_lhd(
            ddg_array, ref_energy, chem_affinities, dna_conc, prot_conc)
        print chem_affinities
        print ref_energy
        print ddg_array
        ref_energy_grad, ddg_grad, chem_affinity_grad = calc_grad(
            ddg_array, ref_energy, chem_affinities, dna_conc, prot_conc)
        
        print chem_affinity_grad
        print ref_energy_grad
        print ddg_grad
        ref_energy += ref_energy_grad.clip(-1,1)
        #chem_affinities += chem_affinity_grad.clip(-1, 1)
        ddg_array += ddg_grad.clip(-1.0, 1.0)
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
