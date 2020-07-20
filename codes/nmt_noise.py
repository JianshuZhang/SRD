'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
import cPickle as pkl
#import ipdb
import numpy
import copy
import pprint
import math
import os
import warnings
import sys
import time

from collections import OrderedDict

from optimizers import adadelta, adam, adadelta_weightnoise
from data_iterator import dataIterator

profile = False

import random
import re

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def conv_norm_weight(kernel_size, nin, nout=None, scale=0.01):
    W = scale * numpy.random.rand(nout, nin, kernel_size, 1)
    return W.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(options, seqs_x, seqs_m, seqs_y, seqs_re, seqs_a, seqs_rea):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y)

    x = numpy.zeros((maxlen_x, n_samples, options['dim_feature'])).astype('float32') # SeqX * batch * dim
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    re = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    a = numpy.zeros((maxlen_y, n_samples, maxlen_x)).astype('float32') # SeqY * batch * SeqX
    rea = numpy.zeros((maxlen_y, n_samples, maxlen_x)).astype('float32') # SeqY * batch * SeqX

    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    re_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    a_mask = numpy.zeros((maxlen_y, n_samples, maxlen_x)).astype('float32')
    rea_mask = numpy.zeros((maxlen_y, n_samples, maxlen_x)).astype('float32')
    
    for idx, [s_x, s_m, s_y, s_re, s_a, s_rea] in enumerate(zip(seqs_x, seqs_m, seqs_y, seqs_re, seqs_a, seqs_rea)):
        x[:lengths_x[idx], idx,:] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1. # the zeros frame is a padding frame to align </s>
        x_mask[:lengths_x[idx], idx] = 1. * s_m
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.
        re[:lengths_y[idx], idx] = s_re
        re_mask[:lengths_y[idx], idx] = 1.
        re_mask[0, idx] = 0. # remove the Start relation
        re_mask[lengths_y[idx]-1, idx] = 0. # remove the End relation
        a[:lengths_y[idx], idx,:lengths_x[idx]] = s_a.T * 1. * s_m[None,:]
        a_mask[:lengths_y[idx], idx, :lengths_x[idx]+1] = 1.
        a_mask[lengths_y[idx]-1, idx, :lengths_x[idx]+1] = 0.
        rea[:lengths_y[idx], idx,:lengths_x[idx]] = s_rea.T * 1. * s_m[None,:]
        rea_mask[:lengths_y[idx], idx, :lengths_x[idx]+1] = 1.
        rea_mask[0, idx, :lengths_x[idx]+1] = 0.

    return x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask

def prepare_valid_data(options, seqs_x, seqs_y, n_words_src=30000, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples, options['dim_feature'])).astype('float32') # SeqX * batch * dim
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx,:] = s_x # the zeros frame is a padding frame to align <eol>
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    Wzrx = numpy.concatenate([norm_weight(nin, dim),
                              norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'Wzrx')] = Wzrx
    params[_p(prefix, 'bzrx')] = numpy.zeros((2 * dim,)).astype('float32')

    Uzrx = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'Uzrx')] = Uzrx

    Whx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Whx')] = Whx
    params[_p(prefix, 'bhx')] = numpy.zeros((dim_nonlin,)).astype('float32')
    Uhx = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Uhx')] = Uhx
    
    # attention
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att
    # coverage conv
    params[_p(prefix, 'conv_Q')] = conv_norm_weight(options['dim_coverage'], 1, dim_nonlin).astype('float32')
    params[_p(prefix, 'conv_Uf')] = norm_weight(dim_nonlin, dimctx)
    params[_p(prefix, 'conv_b')] = numpy.zeros((dimctx,)).astype('float32')
    # attention: context -> attention
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att
    bc_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'bc_att')] = bc_att
    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att
    # alignment loss
    U_align = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_align')] = U_align
    b_align = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'b_align')] = b_align


    # relation decoder params
    reUzrc = numpy.concatenate([ortho_weight(dim_nonlin),
                                ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'reUzrc')] = reUzrc
    params[_p(prefix, 'rebzrc')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')
    reUhc = ortho_weight(dim_nonlin)
    params[_p(prefix, 'reUhc')] = reUhc
    params[_p(prefix, 'rebhc')] = numpy.zeros((dim_nonlin,)).astype('float32')
    Wzrc = numpy.concatenate([norm_weight(dimctx, dim),
                              norm_weight(dimctx, dim)], axis=1)
    params[_p(prefix, 'Wzrc')] = Wzrc
    Whc = norm_weight(dimctx, dim)
    params[_p(prefix, 'Whc')] = Whc

    # related symbol attention
    reW_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'reW_comb_att')] = reW_comb_att
    # coverage conv
    params[_p(prefix, 'conv_reQ')] = conv_norm_weight(options['dim_coverage'], 1, dim_nonlin).astype('float32')
    params[_p(prefix, 'conv_reUf')] = norm_weight(dim_nonlin, dimctx)
    params[_p(prefix, 'conv_reb')] = numpy.zeros((dimctx,)).astype('float32')
    # attention: context -> attention
    reWc_att = norm_weight(dimctx)
    params[_p(prefix, 'reWc_att')] = reWc_att
    rebc_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'rebc_att')] = rebc_att
    # attention:
    reU_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'reU_att')] = reU_att
    rec_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'rec_att')] = rec_att
    # related alignment loss
    reU_align = norm_weight(dimctx, 1)
    params[_p(prefix, 'reU_align')] = reU_align
    reb_align = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'reb_align')] = reb_align


    # last decoder params
    Uzr_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'Uzr_nl')] = Uzr_nl
    params[_p(prefix, 'bzr_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')
    Uh_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Uh_nl')] = Uh_nl
    params[_p(prefix, 'bh_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')
    Wzr_combc = norm_weight(dimctx*2, dim*2)
    params[_p(prefix, 'Wzr_combc')] = Wzr_combc
    Wh_combc = norm_weight(dimctx*2, dim)
    params[_p(prefix, 'Wh_combc')] = Wh_combc

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, pctx_=None, one_step=False,
                   init_memory=None, init_state=None, alpha_past=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Whc')].shape[1]
    dimctx = tparams[_p(prefix, 'Whc')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if alpha_past is None:
        alpha_past = tensor.alloc(0., n_samples, context.shape[0])

    repctx_ = tensor.dot(context, tparams[_p(prefix, 'reWc_att')]) +\
        tparams[_p(prefix, 'rebc_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Whx')]) +\
        tparams[_p(prefix, 'bhx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'Wzrx')]) +\
        tparams[_p(prefix, 'bzrx')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, alpha_past_, rectx_, realpha_, loss_align, loss_realign, pctx_, repctx_, cc_,
                    Uzrx, Uhx, W_comb_att, conv_Q, conv_Uf, conv_b, U_att, c_att, U_align, b_align, 
                    reUzrc, rebzrc, reUhc, rebhc, Wzrc, Whc, reW_comb_att, conv_reQ, conv_reUf, conv_reb, 
                    reU_att, rec_att, reU_align, reb_align, Uzr_nl, bzr_nl, Uh_nl, bh_nl, Wzr_combc, Wh_combc):
        # first GRU
        preact1 = tensor.dot(h_, Uzrx)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)
        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)
        preactx1 = tensor.dot(h_, Uhx)
        preactx1 *= r1
        preactx1 += xx_
        h1 = tensor.tanh(preactx1)
        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        pstate_ = tensor.dot(h1, W_comb_att)
        # converage vector
        cover_F = theano.tensor.nnet.conv2d(alpha_past_[:,None,:,None],conv_Q,border_mode='half') # batch x dim x SeqL x 1
        cover_F = cover_F.dimshuffle(1,2,0,3) # dim x SeqL x batch x 1
        cover_F = cover_F.reshape([cover_F.shape[0],cover_F.shape[1],cover_F.shape[2]])
        assert cover_F.ndim == 3, \
            'Output of conv must be 3-d: #dim x SeqL x batch'
        cover_F = cover_F.dimshuffle(1, 2, 0)
        cover_vector = tensor.dot(cover_F, conv_Uf) + conv_b
        pctx__ = pctx_ + pstate_[None, :, :] + cover_vector
        # attention
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_att
        alpha = alpha - alpha.max()
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / (alpha.sum(0, keepdims=True) + numpy.array(1e-20).astype('float32'))
        # alpha = alpha / (alpha.sum(0, keepdims=True))
        # alignment loss
        loss_align = tensor.dot(pctx__, U_align) + b_align
        loss_align = loss_align.reshape([loss_align.shape[0], loss_align.shape[1]])
        loss_align = tensor.nnet.sigmoid(loss_align)
        alpha_past = alpha_past_ + alpha.T
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)


        # relation decoder
        preact2 = tensor.dot(h1, reUzrc)+rebzrc
        preact2 += tensor.dot(ctx_, Wzrc)
        preact2 = tensor.nnet.sigmoid(preact2)
        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)
        preactx2 = tensor.dot(h1, reUhc)+rebhc
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Whc)
        h2 = tensor.tanh(preactx2)
        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        repstate_ = tensor.dot(h2, reW_comb_att)
        recover_F = theano.tensor.nnet.conv2d(alpha_past[:,None,:,None],conv_reQ,border_mode='half') # batch x dim x SeqL x 1
        recover_F = recover_F.dimshuffle(1,2,0,3) # dim x SeqL x batch x 1
        recover_F = recover_F.reshape([recover_F.shape[0],recover_F.shape[1],recover_F.shape[2]])
        assert recover_F.ndim == 3, \
            'Output of re-conv must be 3-d: #dim x SeqL x batch'
        recover_F = recover_F.dimshuffle(1, 2, 0)
        recover_vector = tensor.dot(recover_F, conv_reUf) + conv_reb
        repctx__ = repctx_ + repstate_[None, :, :] + recover_vector
        # related attention
        repctx__ = tensor.tanh(repctx__)
        realpha = tensor.dot(repctx__, reU_att)+rec_att
        realpha = realpha - realpha.max()
        realpha = realpha.reshape([realpha.shape[0], realpha.shape[1]])
        realpha = tensor.exp(realpha)
        if context_mask:
            realpha = realpha * context_mask
        realpha = realpha / (realpha.sum(0, keepdims=True) + numpy.array(1e-20).astype('float32'))
        # realpha = realpha / (realpha.sum(0, keepdims=True))
        # related alignment loss
        loss_realign = tensor.dot(repctx__, reU_align) + reb_align
        loss_realign = loss_realign.reshape([loss_realign.shape[0], loss_realign.shape[1]])
        loss_realign = tensor.nnet.sigmoid(loss_realign)
        rectx_ = (cc_ * realpha[:, :, None]).sum(0)


        # last decoder
        combine_ctx_ = concatenate([ctx_, rectx_], axis=1)

        preact3 = tensor.dot(h2, Uzr_nl)+bzr_nl
        preact3 += tensor.dot(combine_ctx_, Wzr_combc)
        preact3 = tensor.nnet.sigmoid(preact3)
        r3 = _slice(preact3, 0, dim)
        u3 = _slice(preact3, 1, dim)
        preactx3 = tensor.dot(h2, Uh_nl)+bh_nl
        preactx3 *= r3
        preactx3 += tensor.dot(combine_ctx_, Wh_combc)
        h3 = tensor.tanh(preactx3)
        h3 = u3 * h2 + (1. - u3) * h3
        h3 = m_[:, None] * h3 + (1. - m_)[:, None] * h2

        return h3, ctx_, alpha.T, alpha_past, rectx_, realpha.T, loss_align.T, loss_realign.T

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'Uzrx')],
                   tparams[_p(prefix, 'Uhx')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'conv_Q')],
                   tparams[_p(prefix, 'conv_Uf')],
                   tparams[_p(prefix, 'conv_b')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_att')],
                   tparams[_p(prefix, 'U_align')],
                   tparams[_p(prefix, 'b_align')],
                   tparams[_p(prefix, 'reUzrc')],
                   tparams[_p(prefix, 'rebzrc')],
                   tparams[_p(prefix, 'reUhc')],
                   tparams[_p(prefix, 'rebhc')],
                   tparams[_p(prefix, 'Wzrc')],
                   tparams[_p(prefix, 'Whc')],
                   tparams[_p(prefix, 'reW_comb_att')],
                   tparams[_p(prefix, 'conv_reQ')],
                   tparams[_p(prefix, 'conv_reUf')],
                   tparams[_p(prefix, 'conv_reb')],
                   tparams[_p(prefix, 'reU_att')],
                   tparams[_p(prefix, 'rec_att')],
                   tparams[_p(prefix, 'reU_align')],
                   tparams[_p(prefix, 'reb_align')],
                   tparams[_p(prefix, 'Uzr_nl')],
                   tparams[_p(prefix, 'bzr_nl')],
                   tparams[_p(prefix, 'Uh_nl')],
                   tparams[_p(prefix, 'bh_nl')],
                   tparams[_p(prefix, 'Wzr_combc')],
                   tparams[_p(prefix, 'Wh_combc')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, alpha_past, None, None, None, None, pctx_, repctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                            context.shape[0]),
                                                  tensor.alloc(0., n_samples,
                                                            context.shape[0])],
                                    non_sequences=[pctx_, repctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# Conditional GRU layer with Attention
def param_init_group(options, params, prefix='group', dim=None):

    W = norm_weight(dim, 2)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((2,)).astype('float32')
    params[_p(prefix, 'b')] = b

    return params


def group_layer(tparams, annotation, options, prefix='group'):

    assert annotation.ndim == 3, \
        'annotation must be 3-d: #SeqX x #batch x dimctx'

    nsteps = annotation.shape[0]
    n_samples = annotation.shape[1]
    # dim = annotation.shape[2]

    def _step_slice(a_, g_, a, W, b):

        d = a_[None, :, :] - a
        d = abs(d)
        # d = tensor.tanh(d)
        # d = d * d
        # d = tensor.sqrt(d)
        # d = a_copy * a
        g_ = tensor.dot(d, W) + b
        # g_ = g_.reshape([g_.shape[0], g_.shape[1]])
        # g_ = tensor.nnet.sigmoid(g_)
        g_ = g_ - g_.max()
        g_ = tensor.exp(g_)
        g_ = g_ / (g_.sum(2, keepdims=True) + numpy.array(1e-20).astype('float32'))

        return g_

    seqs = [annotation]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'W')],
                   tparams[_p(prefix, 'b')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(0., nsteps,
                                                            n_samples, 2)],
                                non_sequences=[annotation]+shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb_dec'] = norm_weight(options['dim_target'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r0',
                                              nin=options['dim_feature'],
                                              dim=options['dim_enc'][0])


    hiddenSizes=options['dim_enc']
    for i in range(1,len(hiddenSizes)):
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder'+str(i),
                                                  nin=hiddenSizes[i-1]*2,
                                                  dim=hiddenSizes[i])
        params = get_layer(options['encoder'])[0](options, params,
                                                  prefix='encoder_r'+str(i),
                                                  nin=hiddenSizes[i-1]*2,
                                                  dim=hiddenSizes[i])
    ctxdim = 2 * hiddenSizes[-1]

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim_dec'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim_dec'],
                                              dimctx=ctxdim)

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_gru',
                                nin=options['dim_dec'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word']/2,
                                nout=options['dim_target'])

    # params = get_layer('ff')[0](options, params, prefix='ff_relation_logit_gru',
    #                             nin=options['dim_dec'], nout=options['dim_relation'],
    #                             ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_relation_logit_rectx',
                                nin=ctxdim, nout=options['dim_relation'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_relation_logit_ctx',
                                nin=ctxdim, nout=options['dim_relation'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_relation_logit',
                                nin=options['dim_relation']/2,
                                nout=options['dim_retarget'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.tensor3('x', dtype='float32')
    x_mask_original = tensor.matrix('x_mask_original', dtype='float32')
    a_original = tensor.tensor3('a_original', dtype='float32')
    a_mask_original = tensor.tensor3('a_mask_original', dtype='float32')
    rea_original = tensor.tensor3('rea_original', dtype='float32')
    rea_mask_original = tensor.tensor3('rea_mask_original', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    re = tensor.matrix('re', dtype='int64')
    re_mask = tensor.matrix('re_mask', dtype='float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    x_mask = x_mask_original
    xr_mask = x_mask_original[::-1]

    # n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    # word embedding for forward rnn (source)
    h=x
    hr=xr
    a = a_original
    a_mask = a_mask_original
    rea = rea_original
    rea_mask = rea_mask_original
    hidden_sizes=options['dim_enc']

    for i in range(len(hidden_sizes)):
        proj = get_layer(options['encoder'])[1](tparams, h, options,
                                                prefix='encoder'+str(i),
                                                mask=x_mask)
        # word embedding for backward rnn (source)
        projr = get_layer(options['encoder'])[1](tparams, hr, options,
                                                 prefix='encoder_r'+str(i),
                                                 mask=xr_mask)

        h=concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        if options['down_sample'][i]==1:
            h=h[0::2]
            # x_mask=x_mask[0::2]
            x_mask = pool.pool_2d(x_mask,(2,1),ignore_border=False,stride=(2,1),pad=(0,0),mode='max')
            xr_mask=x_mask[::-1]
            # a: SeqY * batch * SeqX
            a = pool.pool_2d(a,(1,2),ignore_border=False,stride=(1,2),pad=(0,0),mode='max')
            a_mask=a_mask[:,:,0::2]
            rea = pool.pool_2d(rea,(1,2),ignore_border=False,stride=(1,2),pad=(0,0),mode='max')
            rea_mask=rea_mask[:,:,0::2]
        hr=h[::-1]

    # a -- SeqX * batch * SeqY
    # a = a / (tensor.sum(a, axis=0, keepdims=True) + numpy.array(1e-20).astype('float32'))
    # context will be the concatenation of forward and backward rnns
    ctx = h

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    pctx_ = tensor.dot(ctx, tparams[_p('decoder', 'Wc_att')]) +\
        tparams[_p('decoder', 'bc_att')]

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb) # the 0 idx is <eos>!!
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options, prefix='decoder',
                                mask=y_mask, context=ctx, pctx_=pctx_, context_mask=x_mask, 
                                one_step=False, init_state=init_state)
    # proj: h3, ctx_, alpha.T, alpha_past, rectx_, realpha.T, loss_align, loss_realign
    # hidden states of the decoder gru
    proj_h = proj[0]
    # weighted averages of context, generated by attention module
    ctxs = proj[1]
    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2] # SeqY * batch * SeqX while a -- SeqX * batch * SeqY
    # weighted averages of related context, generated by related attention module
    related_ctxs = proj[4]
    # weights (alignment matrix)
    opt_ret['related_alphas'] = proj[5] # SeqY * batch * SeqX
    
    # dec_alphas = proj[2].dimshuffle(2, 1, 0) + numpy.array(1e-20).astype('float32')
    # cost_alphas = - a * tensor.log(dec_alphas) * a_mask
    # cost_alphas = tensor.switch(tensor.isnan(cost_alphas),tensor.zeros_like(cost_alphas),cost_alphas) # in case Nan
    # cost_alphas = tensor.switch(tensor.isinf(cost_alphas),tensor.zeros_like(cost_alphas),cost_alphas)

    loss_align = proj[6] # SeqY * batch * SeqX
    # loss_align = loss_align.dimshuffle(2, 1, 0) # SeqX * batch * SeqY
    loss_align = loss_align + numpy.array(1e-20).astype('float32')
    cost_align = ( - a * tensor.log(loss_align) - (1. - a) * tensor.log(1. - loss_align)) * a_mask
    cost_align = tensor.switch(tensor.isnan(cost_align),tensor.zeros_like(cost_align),cost_align) # in case Nan
    cost_align = tensor.switch(tensor.isinf(cost_align),tensor.zeros_like(cost_align),cost_align)

    loss_realign = proj[7] # SeqY * batch * SeqX
    # loss_realign = loss_realign.dimshuffle(2, 1, 0) # SeqX * batch * SeqY
    loss_realign = loss_realign + numpy.array(1e-20).astype('float32')
    cost_realign = ( - rea * tensor.log(loss_realign) - (1. - rea) * tensor.log(1. - loss_realign)) * rea_mask
    cost_realign = tensor.switch(tensor.isnan(cost_realign),tensor.zeros_like(cost_realign),cost_realign) # in case Nan
    cost_realign = tensor.switch(tensor.isinf(cost_realign),tensor.zeros_like(cost_realign),cost_realign)

    # compute word probabilities
    logit_gru = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx
    # maxout 2
    # maxout layer
    shape = logit.shape
    shape2 = tensor.cast(shape[2] / 2, 'int64')
    shape3 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0],shape[1], shape2, shape3]) # seq*batch*256 -> seq*batch*128*2
    logit=logit.max(3) # seq*batch*128
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, 
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape # (seqL*batch, dim_target)
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]])) # (seqL*batch, dim_target)

    # compute relation probabilities
    # relation_logit_gru = get_layer('ff')[1](tparams, proj_h, options,
    #                                 prefix='ff_relation_logit_gru', activ='linear')
    relation_logit_rectx = get_layer('ff')[1](tparams, related_ctxs, options,
                                    prefix='ff_relation_logit_rectx', activ='linear')
    relation_logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_relation_logit_ctx', activ='linear')
    relation_logit = relation_logit_rectx+relation_logit_ctx
    # maxout 2
    # maxout layer
    reshape1 = relation_logit.shape
    reshape2 = tensor.cast(reshape1[2] / 2, 'int64')
    reshape3 = tensor.cast(2, 'int64')
    relation_logit = relation_logit.reshape([reshape1[0],reshape1[1], reshape2, reshape3]) # seq*batch*256 -> seq*batch*128*2
    relation_logit=relation_logit.max(3) # seq*batch*128
    if options['use_dropout']:
        relation_logit = dropout_layer(relation_logit, use_noise, trng)
    relation_logit = get_layer('ff')[1](tparams, relation_logit, options, 
                               prefix='ff_relation_logit', activ='linear')
    relogit_shp = relation_logit.shape # (seqL*batch, dim_target)
    relation_probs = tensor.nnet.softmax(relation_logit.reshape([relogit_shp[0]*relogit_shp[1],
                                               relogit_shp[2]])) 

    # cost
    cost_pred = tensor.nnet.categorical_crossentropy(probs, y.flatten()) # x is a vector,each value is a 1-of-N position 
    cost_pred = tensor.switch(tensor.isnan(cost_pred),tensor.zeros_like(cost_pred),cost_pred) # in case Nan
    cost_pred = tensor.switch(tensor.isinf(cost_pred),tensor.zeros_like(cost_pred),cost_pred)
    cost_pred = cost_pred.reshape([y.shape[0], y.shape[1]])
    cost_pred = (cost_pred * y_mask).sum(0)

    cost_relation = tensor.nnet.categorical_crossentropy(relation_probs, re.flatten()) # x is a vector,each value is a 1-of-N position 
    cost_relation = tensor.switch(tensor.isnan(cost_relation),tensor.zeros_like(cost_relation),cost_relation) # in case Nan
    cost_relation = tensor.switch(tensor.isinf(cost_relation),tensor.zeros_like(cost_relation),cost_relation)
    cost_relation = cost_relation.reshape([re.shape[0], re.shape[1]])
    cost_relation = (cost_relation * re_mask).sum(0)

    cost_align = options['la'] * cost_align.sum(axis=(0,2))
    cost_realign = options['lb'] * cost_realign.sum(axis=(0,2))

    cost = cost_pred + cost_relation + cost_align + cost_realign

    return trng, use_noise, x, x_mask_original, y, y_mask, re, re_mask, a_original, a_mask_original, rea_original, rea_mask_original, opt_ret, cost, cost_pred, cost_relation, cost_align, cost_realign


# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.tensor3('x', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    h=x
    hr=xr
    h_mask = x_mask
    hr_mask = h_mask[::-1]
    hidden_sizes=options['dim_enc']

    for i in range(len(hidden_sizes)):
        proj = get_layer(options['encoder'])[1](tparams, h, options,
                                                prefix='encoder'+str(i),
                                                mask=h_mask)
        # word embedding for backward rnn (source)
        projr = get_layer(options['encoder'])[1](tparams, hr, options,
                                                 prefix='encoder_r'+str(i),
                                                mask=hr_mask)

        h=concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        if options['down_sample'][i]==1:
            h=h[0::2]
            h_mask = pool.pool_2d(h_mask,(2,1),ignore_border=False,stride=(2,1),pad=(0,0),mode='max')
            hr_mask=h_mask[::-1]
        hr=h[::-1]

    ctx = h
    # get the input for decoder rnn initializer mlp
    # ctx_mean = ctx.mean(0)
    ctx_mean = (ctx * h_mask[:, :, None]).sum(0) / h_mask.sum(0)[:, None]
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx, h_mask]
    inps = [x, x_mask]
    f_init = theano.function(inps, outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    alpha_past = tensor.matrix('alpha_past', dtype='float32')

    pctx_ = tensor.dot(ctx, tparams[_p('decoder', 'Wc_att')]) +\
        tparams[_p('decoder', 'bc_att')]

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options, prefix='decoder',
                                            mask=None, context=ctx, pctx_=pctx_, context_mask=h_mask, 
                                            one_step=True, init_state=init_state,
                                            alpha_past = alpha_past)
    # proj: h3, ctx_, alpha.T (SeqY * batch * SeqX), alpha_past, rectx_, realpha.T, loss_align (SeqY * SeqX * batch), loss_realign
    # get the next hidden state
    next_state = proj[0]
    ctxs = proj[1]
    next_alpha_past = proj[3]
    related_ctxs = proj[4]
    out_align = proj[6] # SeqY * SeqX * batch, SeqY = 1
    # out_align_shape = out_align.shape
    # out_align = out_align.reshape(out_align_shape[-2], out_align_shape[-1]) # SeqX
    out_realign = proj[7] # SeqY * SeqX * batch, SeqY = 1
    # out_realign_shape = out_realign.shape
    # out_realign = out_realign.reshape(out_realign_shape[-2], out_realign_shape[-1])

    logit_gru = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx
    # maxout layer
    shape = logit.shape
    shape1 = tensor.cast(shape[1] / 2, 'int64')
    shape2 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0], shape1, shape2]) # batch*256 -> batch*128*2
    logit=logit.max(2) # batch*500
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)
    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)


    # relation_logit_gru = get_layer('ff')[1](tparams, next_state, options,
    #                                 prefix='ff_relation_logit_gru', activ='linear')
    relation_logit_rectx = get_layer('ff')[1](tparams, related_ctxs, options,
                                    prefix='ff_relation_logit_rectx', activ='linear')
    relation_logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_relation_logit_ctx', activ='linear')
    relation_logit = relation_logit_rectx+relation_logit_ctx
    reshape1 = relation_logit.shape
    reshape2 = tensor.cast(reshape1[1] / 2, 'int64')
    reshape3 = tensor.cast(2, 'int64')
    relation_logit = relation_logit.reshape([reshape1[0], reshape2, reshape3]) # seq*batch*256 -> seq*batch*128*2
    relation_logit=relation_logit.max(2) # seq*batch*128
    if options['use_dropout']:
        relation_logit = dropout_layer(relation_logit, use_noise, trng)
    relation_logit = get_layer('ff')[1](tparams, relation_logit, options, 
                               prefix='ff_relation_logit', activ='linear')
    out_relation_probs = tensor.nnet.softmax(relation_logit)
    out_relation = trng.multinomial(pvals=out_relation_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next..',
    inps = [y, ctx, init_state, alpha_past, h_mask]
    outs = [next_probs, next_sample, next_state, next_alpha_past, out_relation_probs, out_relation, out_align, out_realign]
    f_next = theano.function(inps, outs, name='f_next', profile=profile,on_unused_input='ignore')
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, x_mask, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    relation_sample = []
    relation_sample_score = []
    align_list = []
    realign_list = []
    if stochastic:
        sample_score = 0
        relation_sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_relation_samples = [[]] * live_k
    hyp_relation_scores = numpy.zeros(live_k).astype('float32')

    # get initial state of decoder rnn and encoder context
    inp = [x, x_mask]
    ret = f_init(*inp)
    next_state, ctx0, h_mask0 = ret[0], ret[1], ret[2]
    next_w = -1 * numpy.ones((1,)).astype('int64')
    SeqL = x.shape[0]
    hidden_sizes=options['dim_enc']
    for i in range(len(hidden_sizes)):
        if options['down_sample'][i]==1:
            SeqL = math.ceil(SeqL / 2.)
    next_alpha_past = 0.0 * numpy.ones((1, int(SeqL))).astype('float32')

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        h_mask = numpy.tile(h_mask0, [live_k])
        inps = [next_w, ctx, next_state, next_alpha_past, h_mask]
        ret = f_next(*inps)
        # [next_probs, next_sample, next_state, next_alpha_past, out_relation_probs, out_relation, out_align, out_realign]
        next_p, next_w, next_state, next_alpha_past, next_rp, next_r, next_a, next_rea = \
            ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], ret[6], ret[7]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
                nr = next_rp[0].argmax()
            else:
                nw = next_w[0]
                nr = next_r[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            relation_sample.append(nr)
            relation_sample_score += next_rp[0, nr]
            align_list.append(next_a)
            realign_list.append(next_rea)
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            relation_cand_scores = hyp_relation_scores[:, None] - numpy.log(next_rp)
            relation_cand_flat = relation_cand_scores.flatten()
            relation_ranks_flat = relation_cand_flat.argsort()[:(k-dead_k)]
            relation_voc_size = next_rp.shape[1]
            relation_indices = relation_ranks_flat % relation_voc_size
            relation_costs = relation_cand_flat[relation_ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_alpha_past = []
            new_hyp_relation_samples = []
            new_hyp_relation_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_align = []
            new_hyp_realign = []           

            for idx, [ti, wi, rwi] in enumerate(zip(trans_indices, word_indices, relation_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))
                new_hyp_relation_samples.append(hyp_relation_samples[ti]+[rwi])
                new_hyp_relation_scores[idx] = copy.copy(relation_costs[idx])
                if ii == 0:
                    new_hyp_align.append(copy.copy(next_a[ti]))
                    new_hyp_realign.append(copy.copy(next_rea[ti])) 
                else:
                    if idx < len(hyp_align):
                        new_hyp_align.append(numpy.concatenate((hyp_align[idx],copy.copy(next_a[ti]))))
                        new_hyp_realign.append(numpy.concatenate((hyp_realign[idx],copy.copy(next_rea[ti]))))
                    else:
                        new_hyp_align.append(copy.copy(next_a[ti]))
                        new_hyp_realign.append(copy.copy(next_rea[ti]))             

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_alpha_past = []
            hyp_relation_samples = []
            hyp_relation_scores = []
            hyp_align = []
            hyp_realign = []   

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0: # <eol>
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    relation_sample.append(new_hyp_relation_samples[idx])
                    relation_sample_score.append(new_hyp_relation_scores[idx])
                    align_list.append(new_hyp_align[idx])
                    realign_list.append(new_hyp_realign[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_alpha_past.append(new_hyp_alpha_past[idx])
                    hyp_relation_samples.append(new_hyp_relation_samples[idx])
                    hyp_relation_scores.append(new_hyp_relation_scores[idx])
                    hyp_align.append(new_hyp_align[idx])
                    hyp_realign.append(new_hyp_realign[idx])
                    
            hyp_scores = numpy.array(hyp_scores)
            hyp_relation_scores = numpy.array(hyp_relation_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_alpha_past = numpy.array(hyp_alpha_past)
            

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                relation_sample.append(hyp_relation_samples[idx])
                relation_sample_score.append(hyp_relation_scores[idx])
                align_list.append(hyp_align[idx])
                realign_list.append(hyp_realign[idx])

    return sample, sample_score, relation_sample, relation_sample_score, align_list, realign_list


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=False):
    probs = []
    probs_pred = []
    probs_relation = []
    probs_align = []
    probs_realign = []

    n_done = 0

    for x, m, y, re, a, rea in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask = prepare_data(options, x, m, y, re, a, rea)

        pprobs, pprobs_pred, pprobs_relation, pprobs_align, pprobs_realign = f_log_probs(x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask)
        for pp in pprobs:
            probs.append(pp)
        for pp in pprobs_pred:
            probs_pred.append(pp)
        for pp in pprobs_relation:
            probs_relation.append(pp)
        for pp in pprobs_align:
            probs_align.append(pp)
        for pp in pprobs_realign:
            probs_realign.append(pp)

        if numpy.isnan(numpy.mean(probs_pred)):
            #ipdb.set_trace()
            print 'probs_pred nan'
        if numpy.isnan(numpy.mean(probs_relation)):
            #ipdb.set_trace()
            print 'probs_relation nan'
        if numpy.isnan(numpy.mean(probs_align)):
            #ipdb.set_trace()
            print 'probs_align nan'
        if numpy.isnan(numpy.mean(probs_realign)):
            #ipdb.set_trace()
            print 'probs_realign nan'
        if numpy.isnan(numpy.mean(probs)):
            #ipdb.set_trace()
            print 'probs nan'
        

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs), numpy.array(probs_pred), numpy.array(probs_relation), numpy.array(probs_align), numpy.array(probs_realign)


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    return lexicon

def load_norm(normFile):
    fp=open(normFile)
    name=pkl.load(fp)
    fea_mean=pkl.load(fp)
    name=pkl.load(fp)
    fea_std=pkl.load(fp)
    fp.close()

    return fea_mean, fea_std



def apply_adative_noise(tparams,options,trng,grads,num_examples):
    log_sigma_scale = 2048.0
    init_sigma = 1.0e-12
    model_cost_coefficient=options['model_cost_coeff']
    p_noisy=[]
    Beta=[]
    tparams_p_u=OrderedDict()
    tparams_p_ls2=OrderedDict()
    #tparams_p_s2=OrderedDict()
    for k,p in tparams.iteritems():
        p_u = theano.shared(p.get_value(),name='%s_miu' % k)  # miu
        p_ls2 = theano.shared((numpy.zeros_like(p.get_value()) +
                               numpy.log(init_sigma) * 2. / log_sigma_scale
                               ).astype(dtype=numpy.float32) ,name='%s_sigma' % k) # log_(sigma^2)
        p_s2 = tensor.exp(p_ls2 * log_sigma_scale) # sigma^2

        Beta.append((p_u,p_ls2,p_s2))
        tparams_p_u[k]=p_u
        tparams_p_ls2[k]=p_ls2
        #tparams_p_s2[k]=p_s2
        #p_noisy_tmp = p_u + trng.normal(size=p.get_value().shape) * tensor.sqrt(p_s2)
        #p_noisy.append(p_noisy_tmp)
    


    #  compute the prior mean and variation
    temp_sum = 0.0
    temp_param_count = 0.0
    for p_u, unused_p_ls2, unused_p_s2 in Beta:
        temp_sum = temp_sum + p_u.sum()
        temp_param_count = temp_param_count + p_u.shape.prod()

    prior_u = tensor.cast(temp_sum / temp_param_count, 'float32')

    temp_sum = 0.0
    for p_u, unused_ls2, p_s2 in Beta:
        temp_sum = temp_sum + (p_s2).sum() + (((p_u-prior_u)**2).sum())

    prior_s2 = tensor.cast(temp_sum/temp_param_count, 'float32')
    

    # update miu and sigma gradient with w's grads ## grads is a list
    new_grads_miu = []
    new_grads_sigma = []

    # Warning!!! ????? maybe error????
    # This only works for batch size 1 (we want that the sum of squares
    # be the square of the sum!
    #
    #diag_hessian_estimate = [g**2 for g in grads]

    for (p_u, p_ls2, p_s2),p_grad in zip(Beta,grads):
        p_u_grad = (model_cost_coefficient * (p_u - prior_u) /
                    (num_examples*prior_s2) + p_grad)

        p_ls2_grad = (numpy.float32(model_cost_coefficient *
                                    0.5 / num_examples * log_sigma_scale) *
                      (p_s2/prior_s2 - 1.0) +
                      (0.5*log_sigma_scale) * p_s2 * (p_grad**2)
                      )
        new_grads_miu.append(p_u_grad)
        new_grads_sigma.append(p_ls2_grad)



    # return

    # add noise to weight
    p_add_noise=[]
    for p,p_ls2 in zip(itemlist(tparams),itemlist(tparams_p_ls2)):
        p_add_noise.append((p,p+trng.normal(size=p.get_value().shape) * tensor.sqrt(tensor.exp(p_ls2 * log_sigma_scale))))
    f_apply_noise_to_weight = theano.function([], [], updates=p_add_noise, profile=profile)



    # restore weight
    copy_weight=[]
    for p,p_u in zip(itemlist(tparams),itemlist(tparams_p_u)):
        copy_weight.append((p,p_u))
    f_copy_weight = theano.function([],[],updates=copy_weight, profile=profile)



    return f_apply_noise_to_weight,f_copy_weight,new_grads_miu,new_grads_sigma,tparams_p_u,tparams_p_ls2 #,tparams_p_s2



def train(dim_word=100,  # word vector dimensionality
          dim_relation=100,  # relation vector dimensionality
          dim_enc=1000,  # the number of LSTM units
          dim_dec=1000,  # the number of LSTM units
          dim_coverage=121,
          dim_attention=512,
          dim_reattention=512,
          down_sample=0,
          encoder='gru',
          decoder='gru_cond',
          patience=4,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          model_cost_coeff=0.1,
          lrate=1e-8,  # learning rate
          la=1e-3,  # lambda alignment
          lb=1e-3,  # lambda related alignment
          dim_target=62,  # source vocabulary size
          dim_retarget=62,  # relation vocabulary size
          dim_feature=123,  # target vocabulary size
          max_xlen=100,  # maximum length of the points
          max_ylen=100,  # maximum length of the latex
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          validStart=10,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=['feature.pkl',
                    'mask.pkl',
                    'label.pkl',
                    'align.pkl',
                    'realign.pkl'],
          valid_datasets=['feature_valid.pkl', 
                          'mask_valid.pkl', 
                          'label_valid.pkl',
                          'align_valid.pkl',
                          'realign_valid.pkl'],
          dictionaries=['lexicon.txt',
                        'relexicon.txt'],
          valid_output=['sym_re_path',
                        'ali_path',
                        'reali_path'],
          valid_result=['result.txt'],
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = load_dict(dictionaries[0])
    print 'total chars',len(worddicts)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    reworddicts = load_dict(dictionaries[1])
    print 'total relations',len(reworddicts)
    reworddicts_r = [None] * len(reworddicts)
    for kk, vv in reworddicts.iteritems():
        reworddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    train,train_uid_list = dataIterator(datasets[0], datasets[1], datasets[2], datasets[3], datasets[4], 
                         worddicts, reworddicts, batch_size=batch_size,max_xlen=max_xlen,max_ylen=max_ylen)
    valid,valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2],
                         valid_datasets[3], valid_datasets[4], worddicts, reworddicts, 
                         batch_size=batch_size,max_xlen=max_xlen,max_ylen=max_ylen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)
    tparams = init_tparams(params)

    trng, use_noise, \
    x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask, \
    opt_ret, cost, cost_pred, cost_relation, cost_align, cost_realign = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask]

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    inps_pred = [x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask]
    cost_valid = [cost, cost_pred, cost_relation, cost_align, cost_realign]
    f_log_probs = theano.function(inps_pred, cost_valid, profile=profile)
    print 'Done'

    cost = cost.mean()
    cost_pred = cost_pred.mean()
    cost_relation = cost_relation.mean()
    cost_align = cost_align.mean()
    cost_realign = cost_realign.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    # print 'Building f_cost...',
    # inps2 = [x, x_mask, y, y_mask, pa, a_mask, g, g_mask]
    # out_costs = [cost_original, cost_alpha, cost_group]
    # f_cost = theano.function(inps2, out_costs, profile=profile)
    # print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # apply_adaptive_noise and update gradient for adadelta
    f_apply_noise_to_weight,f_copy_weight,new_grads_miu,new_grads_sigma, \
                tparams_p_u,tparams_p_ls2=apply_adative_noise(tparams,model_options,trng,grads,2*8835)

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizer for miu and sigma',
    cost_all = [cost, cost_pred, cost_relation, cost_align, cost_realign]
    f_grad_shared, f_update_miu, f_update_sigma = eval(optimizer)(lr, tparams_p_u, tparams_p_ls2, new_grads_miu, new_grads_sigma, inps, cost_all)
    print 'Done'
    
    # print model parameters
    print "Model params:\n{0}".format(
            pprint.pformat(sorted([p for p in params])))
    # end



    print 'Optimization'

    history_errs = []
    # reload history
    # if reload_ and os.path.exists(saveto):
    #     history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)

    uidx = 0
    estop = False
    halfLrFlag = 0
    bad_counter = 0
    ud_s = 0
    ud_epoch = 0
    cost_s = 0.
    cost_pred_s = 0.
    cost_relation_s = 0.
    cost_align_s = 0.
    cost_realign_s = 0.
    for eidx in xrange(max_epochs):
        n_samples = 0

        random.shuffle(train) # shuffle data
        ud_epoch_start = time.time()

        for x, m, y, re, a, rea in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            ud_start = time.time()

            x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask = \
                                prepare_data(model_options, x, m, y, re, a, rea)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            f_apply_noise_to_weight()
            # compute cost, grads and copy grads to shared variables
            cost_all = f_grad_shared(x, x_mask, y, y_mask, re, re_mask, a, a_mask, rea, rea_mask)
            cost = cost_all[0]
            cost_s += cost
            cost_pred = cost_all[1]
            cost_pred_s += cost_pred
            cost_relation = cost_all[2]
            cost_relation_s += cost_relation
            cost_align = cost_all[3]
            cost_align_s += cost_align
            cost_realign = cost_all[4]
            cost_realign_s += cost_realign

            # do the update on parameters
            f_update_miu(lrate) # update p_u
            f_update_sigma(lrate) # update p_sigma
            f_copy_weight()

            ud = time.time() - ud_start
            ud_s += ud
            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud_s /= 60.
                cost_s /= dispFreq
                cost_pred_s /= dispFreq
                cost_relation_s /= dispFreq
                cost_align_s /= dispFreq
                cost_realign_s /= dispFreq
                print 'Epoch', eidx, ' Update', uidx, ' Cost_pred %.7f, Cost_re %.7f, Cost_a %.7f, Cost_rea %.7f' % \
                (numpy.float(cost_pred_s),numpy.float(cost_relation_s),numpy.float(cost_align_s),numpy.float(cost_realign_s)), \
                ' UD %.3f' % ud_s, ' epson',lrate, ' bad_counter', bad_counter
                ud_s = 0
                cost_s = 0.
                cost_pred_s = 0.
                cost_relation_s = 0.
                cost_align_s = 0.
                cost_realign_s = 0.

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0 and eidx >= validStart:
                # FIXME: random selection?
                # fpp_sample=open(valid_output[0],'w')
                valid_out_path = valid_output[0]
                valid_ali_path = valid_output[1]
                valid_reali_path = valid_output[2]
                if not os.path.exists(valid_out_path):
                    os.mkdir(valid_out_path)
                if not os.path.exists(valid_ali_path):
                    os.mkdir(valid_ali_path)
                if not os.path.exists(valid_reali_path):
                    os.mkdir(valid_reali_path)

                valid_count_idx=0
                for x, m, y, re, a, rea in valid:
                    for xx, mm in zip(x,m):
                        xx_pad = numpy.zeros((xx.shape[0]+1,xx.shape[1]), dtype='float32')
                        xx_pad[:xx.shape[0],:] = xx
                        mm_pad = numpy.ones((mm.shape[0]+1), dtype='float32')
                        mm_pad[:mm.shape[0]] = mm
                        stochastic = False
                        sample, score, relation_sample, relation_score, align_list, realign_list = \
                                        gen_sample(tparams, f_init, f_next,
                                                   xx_pad[:, None, :], mm_pad[:, None], 
                                                   model_options, trng=trng, k=3,
                                                   maxlen=max_ylen,
                                                   stochastic=stochastic,
                                                   argmax=False)
                        if stochastic:
                            ss = sample
                            rs = relation_sample
                        else:
                            score = score / numpy.array([len(s) for s in sample])
                            relation_score = relation_score / numpy.array([len(r) for r in relation_sample])
                            min_score_index = (score+relation_score).argmin()
                            ss = sample[min_score_index]
                            rs = relation_sample[min_score_index]
                            ali = align_list[min_score_index]
                            reali = realign_list[min_score_index]
                        fpp_sample = open(valid_out_path+valid_uid_list[valid_count_idx]+'.txt','w')
                        file_align_sample = valid_ali_path+valid_uid_list[valid_count_idx]+'_align.txt'
                        file_realign_sample = valid_reali_path+valid_uid_list[valid_count_idx]+'_realign.txt'
                        # fpp_sample.write(valid_uid_list[valid_count_idx])
                        valid_count_idx=valid_count_idx+1
                        # for vv in ss:
                        #     if vv == 0: # <eol>
                        #         break
                        #     fpp_sample.write(' '+worddicts_r[vv])
                        # fpp_sample.write('\n')
                        for vv, rv in zip(ss, rs):
                            string = worddicts_r[vv] + '\t' + reworddicts_r[rv] + '\n'
                            fpp_sample.write(string)
                            if vv == 0:
                                break
                        ali_shape = [len(ss),len(ali)/len(ss)]
                        ali = ali.reshape(ali_shape)
                        reali = reali.reshape(ali_shape)
                        numpy.savetxt(file_align_sample, ali)
                        numpy.savetxt(file_realign_sample, reali)
                        fpp_sample.close()
                # print 'valid set decode done'
                # ud_epoch = (time.time() - ud_epoch_start) / 60.
                # print 'epoch cost time ... ', ud_epoch



            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0 and eidx >= validStart:
                use_noise.set_value(0.)
                valid_errs_cost, valid_errs_pred, valid_errs_relation, valid_errs_align, valid_errs_realign = \
                                        pred_probs(f_log_probs, prepare_data, model_options, valid)
                valid_err_cost = valid_errs_cost.mean()
                valid_err_pred = valid_errs_pred.mean()
                valid_err_relation = valid_errs_relation.mean()
                valid_err_align = valid_errs_align.mean()
                valid_err_realign = valid_errs_realign.mean()

                print 'valid set decode done'
                ud_epoch = (time.time() - ud_epoch_start) / 60.
                print 'epoch cost time ... ', ud_epoch

                # compute wer
                label_files_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/label_r1/test/'
                os.system('python compute_sym-re_cer.py ' + valid_output[0] + ' ' + label_files_path + ' ' + valid_result[0])
                fpp=open(valid_result[0])
                lines = fpp.readlines()
                fpp.close()
                part1 = lines[-2].split()
                if part1[0] == 'CER':
                    valid_cer=100. * float(part1[1])
                else:
                    print 'no CER result'
                part2 = lines[-1].split()
                if part2[0] == 'reCER':
                    valid_recer=100. * float(part2[1])
                else:
                    print 'no reCER result'
                os.system('python evaluate_ExpRate.py ./result/')
                sfpp=open('./result/test_ExpRate_result.txt')
                slines = sfpp.readlines()
                sfpp.close()
                sparts = slines[0].split()
                valid_sacc = float(sparts[1])
                valid_err=0.6*(valid_cer+1.*valid_recer)+0.4*(100.-valid_sacc)
                # valid_err=valid_err_cost
                history_errs.append(valid_err)

                if uidx/validFreq == 0 or valid_err <= numpy.array(history_errs).min(): # the first time valid or worse model
                    best_p = unzip(tparams)
                    bad_counter = 0

                if uidx/validFreq != 0 and valid_err > numpy.array(history_errs).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag==2:
                            print 'Early Stop!'
                            estop = True
                            break
                        else:
                            print 'Lr decay and retrain!'
                            bad_counter = 0
                            lrate /= 10.
                            params = best_p
                            halfLrFlag += 1

                if numpy.isnan(valid_err):
                    #ipdb.set_trace()
                    print 'valid_err nan'

                print 'Valid CER: %.2f%%, relation_CER: %.2f%%, ExpRate: %.2f%%' % (valid_cer,valid_recer,valid_sacc)
                print 'Valid Cost: %.4f, Cost_pred: %.4f, Cost_relation: %.4f, Cost_align: %.4f, cost_realign: %.4f' \
                    % (valid_err_cost, valid_err_pred, valid_err_relation, valid_err_align, valid_err_realign)

            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break
    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)

    valid_err, valid_err_pred, valid_err_relation, valid_err_align, valid_err_realign = \
                         pred_probs(f_log_probs, prepare_data, model_options, valid)
    valid_err = valid_err.mean()
    print 'Valid ', valid_err

    # params = copy.copy(best_p)
    # numpy.savez(saveto, zipped_params=best_p,
    #             history_errs=history_errs,
    #             **params)

    return valid_err


if __name__ == '__main__':
    pass
