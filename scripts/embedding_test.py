import timeit

timer = timeit.Timer(
    "calc_affinities(seqs, ddg)",
    setup="""
import numpy as np

import theano
import theano.tensor as TT
from theano.tensor.signal.conv import conv2d as theano_conv2d


one_hot_seqs = TT.tensor3(name='one_hot_seqs', dtype='float32')
ddg = TT.matrix(name='ddg', dtype='float32')
bs_affinities = theano_conv2d(one_hot_seqs, ddg)

calc_affinities = theano.function(
    [one_hot_seqs, ddg], 
    bs_affinities.min(1) )

seqs = np.zeros((100000, 100, 4), dtype='float32')
ddg = np.zeros((9, 4), dtype='float32')
""")
#print timer.timeit(number=1)

import numpy as np

import theano
import theano.tensor as TT
from theano.tensor.nnet.conv import conv2d as theano_conv2d
from theano.tensor.extra_ops import to_one_hot
from pyDNAbinding.sequence import one_hot_encode_sequences


def one_hot(t, r=None):
    """
    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.
    
    if r is not given, r is set to max(t) + 1
    """
    if r is None:
        r = TT.max(t) + 1
        
    ranges = TT.shape_padleft(TT.arange(r), t.ndim)
    return TT.eq(ranges, TT.shape_padright(t, 1))
                            
def build_embedding_conv(word_len):
    rv = []
    for i in xrange(word_len):
        rv.append([0, (4**i), 2*(4**i), 3*(4**i)])
    return np.array(rv, dtype='float32').T[None,None,::-1,::-1]

def multi_base_embedding(seqs, word_len):
    #first build the embedding matrix
    transform_array = build_embedding_conv(word_len)
    # for each offset
    res = []
    seq_len = seqs.shape[-1]
    num_words = (seq_len-word_len+1)//word_len
    for i in xrange(word_len):
        # build the index matrices
        
        trans_seqs = theano_conv2d(
            seqs[:,:,:,i:i+num_words*word_len],
            transform_array,
            border_mode='valid',
            subsample=(1,word_len)
        )
        #res.append(trans_seqs)
        # reconvert to one-hot
        res.append(one_hot(trans_seqs[:,0,:,:], 4**word_len))
    return TT.concatenate(res, axis=2).dimshuffle(0,1,3,2)

word_len = 2
#transform_array = build_embedding_conv(word_len)
#print transform_array
#trans_seqs = theano_conv2d(
#    one_hot_seqs, transform_array, border_mode='valid', subsample=(word_len,1))
seqs = ['TATGGGTT', 'AAGGGGTT', 'TTTTTTTT']
#seqs = ['TTTT',]
one_hot_seqs = TT.tensor4(name='one_hot_seqs', dtype='float32')
f = theano.function(
    [one_hot_seqs,], multi_base_embedding(one_hot_seqs, word_len))

#ddg = TT.matrix(name='ddg', dtype='float32')
#bs_affinities = theano_conv2d(one_hot_seqs, ddg)
#
#calc_affinities = theano.function(
#    [one_hot_seqs, ddg], 
#    bs_affinities )

#seqs = np.zeros((10, 10, 4), dtype='float32')
one_hot_seqs = np.swapaxes(one_hot_encode_sequences(seqs), 1, 2)[:,None,:,:]
print seqs
print one_hot_seqs.shape
print f(one_hot_seqs)
print one_hot_seqs
print one_hot_seqs.shape
print f(one_hot_seqs).shape
