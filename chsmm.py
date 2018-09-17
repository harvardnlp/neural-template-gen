import sys
import os
import math
import random
import argparse
from collections import defaultdict, Counter
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import labeled_data
from utils import logsumexp1, make_fwd_constr_idxs, make_bwd_constr_idxs, backtrace3, backtrace
from data.utils import get_wikibio_poswrds, get_e2e_poswrds
import infc


class HSMM(nn.Module):
    """
    standard hsmm
    """
    def __init__(self, wordtypes, gentypes, opt):
        super(HSMM, self).__init__()
        self.K = opt.K
        self.Kmul = opt.Kmul
        self.L = opt.L
        self.A_dim = opt.A_dim
        self.unif_lenps = opt.unif_lenps
        self.A_from = nn.Parameter(torch.Tensor(opt.K*opt.Kmul, opt.A_dim))
        self.A_to = nn.Parameter(torch.Tensor(opt.A_dim, opt.K*opt.Kmul))
        if self.unif_lenps:
            self.len_scores = nn.Parameter(torch.ones(1, opt.L))
            self.len_scores.requires_grad = False
        else:
            self.len_decoder = nn.Linear(2*opt.A_dim, opt.L)

        self.yes_self_trans = opt.yes_self_trans
        if not self.yes_self_trans:
            selfmask = torch.Tensor(opt.K*opt.Kmul).fill_(-float("inf"))
            self.register_buffer('selfmask', Variable(torch.diag(selfmask), requires_grad=False))

        self.max_pool = opt.max_pool
        self.emb_size, self.layers, self.hid_size = opt.emb_size, opt.layers, opt.hid_size
        self.pad_idx = opt.pad_idx
        self.lut = nn.Embedding(wordtypes, opt.emb_size, padding_idx=opt.pad_idx)
        self.mlpinp = opt.mlpinp
        self.word_ar = opt.word_ar
        self.ar = False
        inp_feats = 4
        sz_mult = opt.mlp_sz_mult
        if opt.mlpinp:
            rnninsz = sz_mult*opt.emb_size
            mlpinp_sz = inp_feats*opt.emb_size
            self.inpmlp = nn.Sequential(nn.Linear(mlpinp_sz, sz_mult*opt.emb_size),
                                        nn.ReLU())
        else:
            rnninsz = inp_feats*opt.emb_size

        self.start_emb = nn.Parameter(torch.Tensor(1, 1, rnninsz))
        self.pad_emb = nn.Parameter(torch.zeros(1, 1, rnninsz))

        self.one_rnn = opt.one_rnn
        if opt.one_rnn:
            rnninsz += opt.emb_size

        self.seg_rnns = nn.ModuleList()
        if opt.one_rnn:
            self.seg_rnns.append(nn.LSTM(rnninsz, opt.hid_size,
                                         opt.layers, dropout=opt.dropout))
            self.state_embs = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.emb_size))
        else:
            for _ in xrange(opt.K):
                self.seg_rnns.append(nn.LSTM(rnninsz, opt.hid_size,
                                             opt.layers, dropout=opt.dropout))
        self.ar_rnn = nn.LSTM(opt.emb_size, opt.hid_size, opt.layers, dropout=opt.dropout)

        self.h0_lin = nn.Linear(opt.emb_size, 2*opt.hid_size)
        self.state_att_gates = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))
        self.state_att_biases = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))

        self.sep_attn = opt.sep_attn
        if self.sep_attn:
            self.state_att2_gates = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))
            self.state_att2_biases = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))

        out_hid_sz = opt.hid_size + opt.emb_size
        self.state_out_gates = nn.Parameter(torch.Tensor(opt.K, 1, 1, out_hid_sz))
        self.state_out_biases = nn.Parameter(torch.Tensor(opt.K, 1, 1, out_hid_sz))
        # add one more output word for eop
        self.decoder = nn.Linear(out_hid_sz, gentypes+1)
        self.eop_idx = gentypes
        self.attn_lin1 = nn.Linear(opt.hid_size, opt.emb_size)
        self.linear_out = nn.Linear(opt.hid_size + opt.emb_size, opt.hid_size)

        self.drop = nn.Dropout(opt.dropout)
        self.emb_drop = opt.emb_drop
        self.initrange = opt.initrange
        self.lsm = nn.LogSoftmax(dim=1)
        self.zeros = torch.Tensor(1, 1).fill_(-float("inf")) if opt.lse_obj else torch.zeros(1, 1)
        self.lse_obj = opt.lse_obj
        if opt.cuda:
            self.zeros = self.zeros.cuda()

        # src encoder stuff
        self.src_bias = nn.Parameter(torch.Tensor(1, opt.emb_size))
        self.uniq_bias = nn.Parameter(torch.Tensor(1, opt.emb_size))

        self.init_lin = nn.Linear(opt.emb_size, opt.K*opt.Kmul)
        self.cond_A_dim = opt.cond_A_dim
        self.smaller_cond_dim = opt.smaller_cond_dim
        if opt.smaller_cond_dim > 0:
            self.cond_trans_lin = nn.Sequential(
                nn.Linear(opt.emb_size, opt.smaller_cond_dim),
                nn.ReLU(),
                nn.Linear(opt.smaller_cond_dim, opt.K*opt.Kmul*opt.cond_A_dim*2))
        else:
            self.cond_trans_lin = nn.Linear(opt.emb_size, opt.K*opt.Kmul*opt.cond_A_dim*2)
        self.init_weights()


    def init_weights(self):
        """
        (re)init weights
        """
        initrange = self.initrange
        self.lut.weight.data.uniform_(-initrange, initrange)
        self.lut.weight.data[self.pad_idx].zero_()
        self.lut.weight.data[corpus.dictionary.word2idx["<ncf1>"]].zero_()
        self.lut.weight.data[corpus.dictionary.word2idx["<ncf2>"]].zero_()
        self.lut.weight.data[corpus.dictionary.word2idx["<ncf3>"]].zero_()
        params = [self.src_bias, self.state_out_gates, self.state_att_gates,
                  self.state_out_biases, self.state_att_biases, self.start_emb,
                  self.A_from, self.A_to, self.uniq_bias]
        if self.sep_attn:
            params.extend([self.state_att2_gates, self.state_att2_biases])
        if self.one_rnn:
            params.append(self.state_embs)

        for param in params:
            param.data.uniform_(-initrange, initrange)

        rnns = [rnn for rnn in self.seg_rnns]
        rnns.append(self.ar_rnn)
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)

        lins = [self.init_lin, self.decoder, self.attn_lin1, self.linear_out, self.h0_lin]
        if self.smaller_cond_dim == 0:
            lins.append(self.cond_trans_lin)
        else:
            lins.extend([self.cond_trans_lin[0], self.cond_trans_lin[2]])
        if not self.unif_lenps:
            lins.append(self.len_decoder)
        if self.mlpinp:
            lins.append(self.inpmlp[0])
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if lin.bias is not None:
                lin.bias.data.zero_()


    def trans_logprobs(self, uniqenc, seqlen):
        """
        args:
          uniqenc - bsz x emb_size
        returns:
          1 x K tensor and seqlen-1 x bsz x K x K tensor of log probabilities,
                           where lps[i] is p(q_{i+1} | q_i)
        """
        bsz = uniqenc.size(0)
        K = self.K*self.Kmul
        A_dim = self.A_dim
        # bsz x K*A_dim*2 -> bsz x K x A_dim or bsz x K x 2*A_dim
        cond_trans_mat = self.cond_trans_lin(uniqenc).view(bsz, K, -1)
        # nufrom, nuto each bsz x K x A_dim
        A_dim = self.cond_A_dim
        nufrom, nuto = cond_trans_mat[:, :, :A_dim], cond_trans_mat[:, :, A_dim:]
        A_from, A_to = self.A_from, self.A_to
        if self.drop.p > 0:
            A_from = self.drop(A_from)
            nufrom = self.drop(nufrom)
        tscores = torch.mm(A_from, A_to)
        if not self.yes_self_trans:
            tscores = tscores + self.selfmask
        trans_lps = tscores.unsqueeze(0).expand(bsz, K, K)
        trans_lps = trans_lps + torch.bmm(nufrom, nuto.transpose(1, 2))
        trans_lps = self.lsm(trans_lps.view(-1, K)).view(bsz, K, K)

        init_lps = self.lsm(self.init_lin(uniqenc)) # bsz x K
        return init_lps, trans_lps.view(1, bsz, K, K).expand(seqlen-1, bsz, K, K)

    def len_logprobs(self):
        """
        returns:
           [1xK tensor, 2 x K tensor, .., L-1 x K tensor, L x K tensor] of logprobs
        """
        K = self.K*self.Kmul
        state_embs = torch.cat([self.A_from, self.A_to.t()], 1) # K x 2*A_dim
        if self.unif_lenps:
            len_scores = self.len_scores.expand(K, self.L)
        else:
            len_scores = self.len_decoder(state_embs) # K x L
        lplist = [Variable(len_scores.data.new(1, K).zero_())]
        for l in xrange(2, self.L+1):
            lplist.append(self.lsm(len_scores.narrow(1, 0, l)).t())
        return lplist, len_scores


    def to_seg_embs(self, xemb):
        """
        xemb - bsz x seqlen x emb_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [1 2 3 4]  becomes [<s> <s> <s> <s> <s> <s> <s> <s>]
                 [5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                    [ 2   3   4  <p>  6   7   8  <p>]
                                    [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlen, emb_size = xemb.size()
        newx = [self.start_emb.expand(bsz, seqlen, emb_size)]
        newx.append(xemb)
        for i in xrange(1, self.L):
            pad = self.pad_emb.expand(bsz, i, emb_size)
            rowi = torch.cat([xemb[:, i:], pad], 1)
            newx.append(rowi)
        # L+1 x bsz x seqlen x emb_size -> L+1 x bsz*seqlen x emb_size
        return torch.stack(newx).view(self.L+1, -1, emb_size)


    def to_seg_hist(self, states):
        """
        states - bsz x seqlen+1 x rnn_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [<b> 1 2 3 4]  becomes [<b>  1   2   3  <b>  5   6   7 ]
                 [<b> 5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                        [ 2   3   4  <p>  6   7   8  <p>]
                                        [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlenp1, rnn_size = states.size()
        newh = [states[:, :seqlenp1-1, :]] # [bsz x seqlen x rnn_size]
        newh.append(states[:, 1:, :])
        for i in xrange(1, self.L):
            pad = self.pad_emb[:, :, :rnn_size].expand(bsz, i, rnn_size)
            rowi = torch.cat([states[:, i+1:, :], pad], 1)
            newh.append(rowi)
        # L+1 x bsz x seqlen x rnn_size -> L+1 x bsz*seqlen x rnn_size
        return torch.stack(newh).view(self.L+1, -1, rnn_size)


    def obs_logprobs(self, x, srcenc, srcfieldenc, fieldmask, combotargs, bsz):
        """
        args:
          x - seqlen x bsz x max_locs x nfeats
          srcenc - bsz x emb_size
          srcfieldenc - bsz x nfields x dim
          fieldmask - bsz x nfields mask with 0s and -infs where it's a dummy field
          combotargs - L x bsz*seqlen x max_locs
        returns:
          a L x seqlen x bsz x K tensor, where l'th row has prob of sequences of length l+1.
          specifically, obs_logprobs[:,t,i,k] gives p(x_t|k), p(x_{t:t+1}|k), ..., p(x_{t:t+l}|k).
          the infc code ignores the entries rows corresponding to x_{t:t+m} where t+m > T
        """
        seqlen, bsz, maxlocs, nfeats = x.size()
        embs = self.lut(x.view(seqlen, -1)) # seqlen x bsz*maxlocs*nfeats x emb_size

        if self.mlpinp:
            inpembs = self.inpmlp(embs.view(seqlen, bsz, maxlocs, -1)).mean(2)
        else:
            inpembs = embs.view(seqlen, bsz, maxlocs, -1).mean(2) # seqlen x bsz x nfeats*emb_size

        if self.emb_drop:
            inpembs = self.drop(inpembs)

        if self.ar:
            if self.word_ar:
                ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, 0, 0] # seqlen x bsz x embsz
            else: # ar on fields
                ar_embs = embs.view(seqlen, bsz, maxlocs, nfeats, -1)[:, :, :, 1].mean(2) # same
            if self.emb_drop:
                ar_embs = self.drop(ar_embs)

            # add on initial <bos> thing; this is a HACK!
            embsz = ar_embs.size(2)
            ar_embs = torch.cat([self.lut.weight[2].view(1, 1, embsz).expand(1, bsz, embsz),
                                    ar_embs], 0) # seqlen+1 x bsz x emb_size
            ar_states, _ = self.ar_rnn(ar_embs) # seqlen+1 x bsz x rnn_size

        # get L+1 x bsz*seqlen x emb_size segembs
        segembs = self.to_seg_embs(inpembs.transpose(0, 1))
        Lp1, bszsl, _ = segembs.size()
        if self.ar:
            segars = self.to_seg_hist(ar_states.transpose(0, 1)) #L+1 x bsz*seqlen x rnn_size

        bsz, nfields, encdim = srcfieldenc.size()
        layers, rnn_size = self.layers, self.hid_size

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        inits = self.h0_lin(srcenc) # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:] # (bsz x dim, bsz x dim)
        h0 = F.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()

        # easiest to just loop over K
        state_emb_sz = self.state_embs.size(3)
        seg_lls = []
        for k in xrange(self.K):
            if self.one_rnn:
                condembs = torch.cat(
                    [segembs, self.state_embs[k].expand(Lp1, bszsl, state_emb_sz)], 2)
                states, _ = self.seg_rnns[0](condembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size
            else:
                states, _ = self.seg_rnns[k](segembs, (h0, c0)) # L+1 x bsz*seqlen x rnn_size

            if self.ar:
                states = states + segars # L+1 x bsz*seqlen x rnn_size

            if self.drop.p > 0:
                states = self.drop(states)
            attnin1 = (states * self.state_att_gates[k].expand_as(states)
                       + self.state_att_biases[k].expand_as(states)).view(
                           Lp1, bsz, seqlen, -1)
            # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x rnn_size
            attnin1 = attnin1.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
            attnin1 = F.tanh(attnin1)
            ascores = torch.bmm(attnin1, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
            ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)
            aprobs = F.softmax(ascores, dim=2)
            # bsz x (L+1)seqlen x nfields * bsz x nfields x dim -> bsz x (L+1)seqlen x dim
            ctx = torch.bmm(aprobs, srcfieldenc)
            # concatenate states and ctx to get L+1 x bsz x seqlen x rnn_size + encdim
            cat_ctx = torch.cat([states.view(Lp1, bsz, seqlen, -1),
                                 ctx.view(bsz, Lp1, seqlen, -1).transpose(0, 1)], 3)
            out_hid_sz = rnn_size + encdim
            cat_ctx = cat_ctx.view(Lp1, -1, out_hid_sz)
            # now linear to get L+1 x bsz*seqlen x rnn_size
            states_k = F.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
                              + self.state_out_biases[k].expand_as(cat_ctx)).view(
                                  Lp1, -1, out_hid_sz)

            if self.sep_attn:
                attnin2 = (states * self.state_att2_gates[k].expand_as(states)
                           + self.state_att2_biases[k].expand_as(states)).view(
                               Lp1, bsz, seqlen, -1)
                # L+1 x bsz x seqlen x rnn_size -> bsz x (L+1)seqlen x emb_size
                attnin2 = attnin2.transpose(0, 1).contiguous().view(bsz, Lp1*seqlen, -1)
                attnin2 = F.tanh(attnin2)
                ascores = torch.bmm(attnin2, srcfieldenc.transpose(1, 2)) # bsz x (L+1)slen x nfield
                ascores = ascores + fieldmask.unsqueeze(1).expand_as(ascores)

            normfn = F.log_softmax if self.lse_obj else F.softmax
            wlps_k = normfn(torch.cat([self.decoder(states_k.view(-1, out_hid_sz)), #L+1*bsz*sl x V
                                       ascores.view(bsz, Lp1, seqlen, nfields).transpose(
                                           0, 1).contiguous().view(-1, nfields)], 1), dim=1)
            # concatenate on dummy column for when only a single answer...
            wlps_k = torch.cat([wlps_k, Variable(self.zeros.expand(wlps_k.size(0), 1))], 1)
            # get scores for predicted next-words (but not for last words in each segment as usual)
            psk = wlps_k.narrow(0, 0, self.L*bszsl).gather(1, combotargs.view(self.L*bszsl, -1))
            if self.lse_obj:
                lls_k = logsumexp1(psk)
            else:
                lls_k = psk.sum(1).log()

            # sum up log probs of words in each segment
            seglls_k = lls_k.view(self.L, -1).cumsum(0) # L x bsz*seqlen
            # need to add end-of-phrase prob too
            eop_lps = wlps_k.narrow(0, bszsl, self.L*bszsl)[:, self.eop_idx] # L*bsz*seqlen
            if self.lse_obj:
                seglls_k = seglls_k + eop_lps.contiguous().view(self.L, -1)
            else:
                seglls_k = seglls_k + eop_lps.log().view(self.L, -1)
            seg_lls.append(seglls_k)

        #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
        obslps = torch.stack(seg_lls).view(self.K, self.L, bsz, -1).transpose(
            0, 3).transpose(0, 1)
        if self.Kmul > 1:
            obslps = obslps.repeat(1, 1, 1, self.Kmul)
        return obslps


    def encode(self, src, avgmask, uniqfields):
        """
        args:
          src - bsz x nfields x nfeats
          avgmask - bsz x nfields, with 0s for pad and 1/tru_nfields for rest
          uniqfields - bsz x maxfields
        returns bsz x emb_size, bsz x nfields x emb_size
        """
        bsz, nfields, nfeats = src.size()
        emb_size = self.lut.embedding_dim
        # do src stuff that depends on words
        embs = self.lut(src.view(-1, nfeats)) # bsz*nfields x nfeats x emb_size
        if self.max_pool:
            embs = F.relu(embs.sum(1) + self.src_bias.expand(bsz*nfields, emb_size))
            if avgmask is not None:
                masked = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size))
            else:
                masked = embs.view(bsz, nfields, emb_size)
            srcenc = F.max_pool1d(masked.transpose(1, 2), nfields).squeeze(2)  # bsz x emb_size
        else:
            embs = F.tanh(embs.sum(1) + self.src_bias.expand(bsz*nfields, emb_size))
            # average it manually, bleh
            if avgmask is not None:
                srcenc = (embs.view(bsz, nfields, emb_size)
                          * avgmask.unsqueeze(2).expand(bsz, nfields, emb_size)).sum(1)
            else:
                srcenc = embs.view(bsz, nfields, emb_size).mean(1) # bsz x emb_size

        srcfieldenc = embs.view(bsz, nfields, emb_size)

        # do stuff that depends only on uniq fields
        uniqenc = self.lut(uniqfields).sum(1) # bsz x nfields x emb_size -> bsz x emb_size

        # add a bias
        uniqenc = uniqenc + self.uniq_bias.expand_as(uniqenc)
        uniqenc = F.relu(uniqenc)

        return srcenc, srcfieldenc, uniqenc

    def get_next_word_dist(self, hid, k, srcfieldenc):
        """
        hid - 1 x bsz x rnn_size
        srcfieldenc - 1 x nfields x dim
        returns a bsz x nthings dist; not a log dist
        """
        bsz = hid.size(1)
        _, nfields, rnn_size = srcfieldenc.size()
        srcfldenc = srcfieldenc.expand(bsz, nfields, rnn_size)
        attnin1 = (hid * self.state_att_gates[k].expand_as(hid)
                   + self.state_att_biases[k].expand_as(hid)) # 1 x bsz x rnn_size
        attnin1 = F.tanh(attnin1)
        ascores = torch.bmm(attnin1.transpose(0, 1), srcfldenc.transpose(1, 2)) # bsz x 1 x nfields
        aprobs = F.softmax(ascores, dim=2)
        ctx = torch.bmm(aprobs, srcfldenc) # bsz x 1 x rnn_size
        cat_ctx = torch.cat([hid, ctx.transpose(0, 1)], 2) # 1 x bsz x rnn_size
        state_k = F.tanh(cat_ctx * self.state_out_gates[k].expand_as(cat_ctx)
                         + self.state_out_biases[k].expand_as(cat_ctx)) # 1 x bsz x rnn_size

        if self.sep_attn:
            attnin2 = (hid * self.state_att2_gates[k].expand_as(hid)
                       + self.state_att2_biases[k].expand_as(hid))
            attnin2 = F.tanh(attnin2)
            ascores = torch.bmm(attnin2.transpose(0, 1), srcfldenc.transpose(1, 2)) # bsz x 1 x nfld

        wlps_k = F.softmax(torch.cat([self.decoder(state_k.squeeze(0)),
                                      ascores.squeeze(1)], 1), dim=1)
        return wlps_k.data

    def collapse_word_probs(self, row2tblent, wrd_dist, corpus):
        """
        wrd_dist is a K x nwords matrix and it gets modified.
        this collapsing only makes sense if src_tbl is the same for every row.
        """
        nout_wrds = self.decoder.out_features
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        # collapse probabilities
        first_seen = {}
        for i, (field, idx, wrd) in row2tblent.iteritems():
            if field is not None:
                if wrd not in first_seen:
                    first_seen[wrd] = i
                    # add gen prob if any
                    if wrd in corpus.genset:
                        widx = w2i[wrd]
                        wrd_dist[:, nout_wrds + i].add_(wrd_dist[:, widx])
                        wrd_dist[:, widx].zero_()
                else: # seen it before, so add its prob
                    wrd_dist[:, nout_wrds + first_seen[wrd]].add_(wrd_dist[:, nout_wrds + i])
                    wrd_dist[:, nout_wrds + i].zero_()
            else: # should really have zeroed out before, but this is easier
                wrd_dist[:, nout_wrds + i].zero_()

    def temp_bs(self, corpus, ss, start_inp, exh0, exc0, srcfieldenc,
                     len_lps, row2tblent, row2feats, K, final_state=False):
        """
        ss - discrete state index
        exh0 - layers x 1 x rnn_size
        exc0 - layers x 1 x rnn_size
        start_inp - 1 x 1 x emb_size
        len_lps - K x L, log normalized
        """
        rul_ss = ss % self.K
        i2w = corpus.dictionary.idx2word
        w2i = corpus.dictionary.word2idx
        genset = corpus.genset
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]
        state_emb_sz = self.state_embs.size(3) if self.one_rnn else 0
        if self.one_rnn:
            cond_start_inp = torch.cat([start_inp, self.state_embs[rul_ss]], 2) # 1 x 1 x cat_size
            hid, (hc, cc) = self.seg_rnns[0](cond_start_inp, (exh0, exc0))
        else:
            hid, (hc, cc) = self.seg_rnns[rul_ss](start_inp, (exh0, exc0))
        curr_hyps = [(None, None)]
        best_wscore, best_lscore = None, None # so we can truly average over words etc later
        best_hyp, best_hyp_score = None, -float("inf")
        curr_scores = torch.zeros(K, 1)
        # N.B. we assume we have a single feature row for each timestep rather than avg
        # over them as at training time. probably better, but could conceivably average like
        # at training time.
        inps = Variable(torch.LongTensor(K, 4), volatile=True)
        for ell in xrange(self.L):
            wrd_dist = self.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu() # K x nwords
            # disallow unks
            wrd_dist[:, unk_idx].zero_()
            if not final_state:
                wrd_dist[:, eos_idx].zero_()
            self.collapse_word_probs(row2tblent, wrd_dist, corpus)
            wrd_dist.log_()
            if ell > 0: # add previous scores
                wrd_dist.add_(curr_scores.expand_as(wrd_dist))
            maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2*K)
            cols = wrd_dist.size(1)
            # we'll break as soon as <eos> is at the top of the beam.
            # this ignores <eop> but whatever
            if top2k[0] == eos_idx:
                final_hyp = backtrace(curr_hyps[0])
                final_hyp.append(eos_idx)
                return final_hyp, maxprobs[0], len_lps[ss][ell]

            new_hyps, anc_hs, anc_cs = [], [], []
            #inps.data.fill_(pad_idx)
            inps.data[:, 1].fill_(w2i["<ncf1>"])
            inps.data[:, 2].fill_(w2i["<ncf2>"])
            inps.data[:, 3].fill_(w2i["<ncf3>"])
            for k in xrange(2*K):
                anc, wrd = top2k[k] / cols, top2k[k] % cols
                # check if any of the maxes are eop
                if wrd == self.eop_idx and ell > 0:
                    # add len score (and avg over num words incl eop i guess)
                    wlenscore = maxprobs[k]/(ell+1) + len_lps[ss][ell-1]
                    if wlenscore > best_hyp_score:
                        best_hyp_score = wlenscore
                        best_hyp = backtrace(curr_hyps[anc])
                        best_wscore, best_lscore = maxprobs[k], len_lps[ss][ell-1]
                else:
                    curr_scores[len(new_hyps)][0] = maxprobs[k]
                    if wrd >= self.decoder.out_features: # a copy
                        tblidx = wrd - self.decoder.out_features
                        inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                    else:
                        inps.data[len(new_hyps)][0] = wrd if i2w[wrd] in genset else unk_idx
                    new_hyps.append((wrd, curr_hyps[anc]))
                    anc_hs.append(hc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                    anc_cs.append(cc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                if len(new_hyps) == K:
                    break
            assert len(new_hyps) == K
            curr_hyps = new_hyps
            if self.lut.weight.data.is_cuda:
                inps = inps.cuda()
            embs = self.lut(inps).view(1, K, -1) # 1 x K x nfeats*emb_size
            if self.mlpinp:
                embs = self.inpmlp(embs) # 1 x K x rnninsz
            if self.one_rnn:
                cond_embs = torch.cat([embs, self.state_embs[rul_ss].expand(1, K, state_emb_sz)], 2)
                hid, (hc, cc) = self.seg_rnns[0](cond_embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
            else:
                hid, (hc, cc) = self.seg_rnns[rul_ss](embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
        # hypotheses of length L still need their end probs added
        # N.B. if the <eos> falls off the beam we could end up with situations
        # where we take an L-length phrase w/ a lower score than 1-word followed by eos.
        wrd_dist = self.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu() # K x nwords
        wrd_dist.log_()
        wrd_dist.add_(curr_scores.expand_as(wrd_dist))
        for k in xrange(K):
            wlenscore = wrd_dist[k][self.eop_idx]/(self.L+1) + len_lps[ss][self.L-1]
            if wlenscore > best_hyp_score:
                best_hyp_score = wlenscore
                best_hyp = backtrace(curr_hyps[k])
                best_wscore, best_lscore = wrd_dist[k][self.eop_idx], len_lps[ss][self.L-1]

        return best_hyp, best_wscore, best_lscore


    def gen_one(self, templt, h0, c0, srcfieldenc, len_lps, row2tblent, row2feats):
        """
        src - 1 x nfields x nfeatures
        h0 - rnn_size vector
        c0 - rnn_size vector
        srcfieldenc - 1 x nfields x dim
        len_lps - K x L, log normalized
        returns a list of phrases
        """
        phrases = []
        tote_wscore, tote_lscore, tokes, segs = 0.0, 0.0, 0.0, 0.0
        #start_inp = self.lut.weight[start_idx].view(1, 1, -1)
        start_inp = self.start_emb
        exh0 = h0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        exc0 = c0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        nout_wrds = self.decoder.out_features
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        for stidx, k in enumerate(templt):
            phrs_idxs, wscore, lscore = self.temp_bs(corpus, k, start_inp, exh0, exc0,
                                                     srcfieldenc, len_lps, row2tblent, row2feats,
                                                     args.beamsz, final_state=(stidx == len(templt)-1))
            phrs = []
            for ii in xrange(len(phrs_idxs)):
                if phrs_idxs[ii] < nout_wrds:
                    phrs.append(i2w[phrs_idxs[ii]])
                else:
                    tblidx = phrs_idxs[ii] - nout_wrds
                    _, _, wordstr = row2tblent[tblidx]
                    if args.verbose:
                        phrs.append(wordstr + " (c)")
                    else:
                        phrs.append(wordstr)
            if phrs[-1] == "<eos>":
                break
            phrases.append(phrs)
            tote_wscore += wscore
            tote_lscore += lscore
            tokes += len(phrs_idxs) + 1 # add 1 for <eop> token
            segs += 1

        return phrases, tote_wscore, tote_lscore, tokes, segs


    def temp_ar_bs(self, templt, row2tblent, row2feats, h0, c0, srcfieldenc, len_lps,  K,
                corpus):
        assert self.unif_lenps # ignoring lenps
        exh0 = h0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        exc0 = c0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        start_inp = self.start_emb
        state_emb_sz = self.state_embs.size(3)
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        genset = corpus.genset
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]

        curr_hyps = [(None, None, None)]
        nfeats = 4
        inps = Variable(torch.LongTensor(K, nfeats), volatile=True)
        curr_scores, curr_lens, nulens = torch.zeros(K, 1), torch.zeros(K, 1), torch.zeros(K, 1)
        if self.lut.weight.data.is_cuda:
            inps = inps.cuda()
            curr_scores, curr_lens, nulens = curr_scores.cuda(), curr_lens.cuda(), nulens.cuda()

        # start ar rnn; hackily use bos_idx
        rnnsz = self.ar_rnn.hidden_size
        thid, (thc, tcc) = self.ar_rnn(self.lut.weight[2].view(1, 1, -1)) # 1 x 1 x rnn_size

        for stidx, ss in enumerate(templt):
            final_state = (stidx == len(templt) - 1)
            minq = [] # so we can compare stuff of different lengths
            rul_ss = ss % self.K

            if self.one_rnn:
                cond_start_inp = torch.cat([start_inp, self.state_embs[rul_ss]], 2) # 1x1x cat_size
                hid, (hc, cc) = self.seg_rnns[0](cond_start_inp, (exh0, exc0)) # 1 x 1 x rnn_size
            else:
                hid, (hc, cc) = self.seg_rnns[rul_ss](start_inp, (exh0, exc0)) # 1 x 1 x rnn_size
            hid = hid.expand_as(thid)
            hc = hc.expand_as(thc)
            cc = cc.expand_as(tcc)

            for ell in xrange(self.L+1):
                new_hyps, anc_hs, anc_cs, anc_ths, anc_tcs = [], [], [], [], []
                inps.data[:, 1].fill_(w2i["<ncf1>"])
                inps.data[:, 2].fill_(w2i["<ncf2>"])
                inps.data[:, 3].fill_(w2i["<ncf3>"])

                wrd_dist = self.get_next_word_dist(hid + thid, rul_ss, srcfieldenc) # K x nwords
                currK = wrd_dist.size(0)
                # disallow unks and eos's
                wrd_dist[:, unk_idx].zero_()
                if not final_state:
                    wrd_dist[:, eos_idx].zero_()
                self.collapse_word_probs(row2tblent, wrd_dist, corpus)
                wrd_dist.log_()
                curr_scores[:currK].mul_(curr_lens[:currK])
                wrd_dist.add_(curr_scores[:currK].expand_as(wrd_dist))
                wrd_dist.div_((curr_lens[:currK]+1).expand_as(wrd_dist))
                maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2*K)
                cols = wrd_dist.size(1)
                # used to check for eos here, but maybe we shouldn't

                for k in xrange(2*K):
                    anc, wrd = top2k[k] / cols, top2k[k] % cols
                    # check if any of the maxes are eop
                    if wrd == self.eop_idx and ell > 0 and (not final_state or curr_hyps[anc][0] == eos_idx):
                        ## add len score (and avg over num words *incl eop*)
                        ## actually ignoring len score for now
                        #wlenscore = maxprobs[k]/(ell+1) # + len_lps[ss][ell-1]
                        #assert not final_state or curr_hyps[anc][0] == eos_idx # seems like should hold...
                        heapitem = (maxprobs[k], curr_lens[anc][0]+1, curr_hyps[anc],
                                    thc.narrow(1, anc, 1), tcc.narrow(1, anc, 1))
                        if len(minq) < K:
                            heapq.heappush(minq, heapitem)
                        else:
                            heapq.heappushpop(minq, heapitem)
                    elif ell < self.L: # only allow non-eop if < L so far
                        curr_scores[len(new_hyps)][0] = maxprobs[k]
                        nulens[len(new_hyps)][0] = curr_lens[anc][0]+1
                        if wrd >= self.decoder.out_features: # a copy
                            tblidx = wrd - self.decoder.out_features
                            inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                        else:
                            inps.data[len(new_hyps)][0] = wrd if i2w[wrd] in genset else unk_idx

                        new_hyps.append((wrd, ss, curr_hyps[anc]))
                        anc_hs.append(hc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_cs.append(cc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_ths.append(thc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                        anc_tcs.append(tcc.narrow(1, anc, 1)) # layers x 1 x rnn_size
                    if len(new_hyps) == K:
                        break

                if ell >= self.L: # don't want to put in eops
                    break

                assert len(new_hyps) == K
                curr_hyps = new_hyps
                curr_lens.copy_(nulens)
                embs = self.lut(inps).view(1, K, -1) # 1 x K x nfeats*emb_size
                if self.word_ar:
                    ar_embs = embs.view(1, K, nfeats, -1)[:, :, 0] # 1 x K x emb_size
                else: # ar on fields
                    ar_embs = embs.view(1, K, nfeats, -1)[:, :, 1] # 1 x K x emb_size
                if self.mlpinp:
                    embs = self.inpmlp(embs) # 1 x K x rnninsz
                if self.one_rnn:
                    cond_embs = torch.cat([embs, self.state_embs[rul_ss].expand(
                        1, K, state_emb_sz)], 2)
                    hid, (hc, cc) = self.seg_rnns[0](cond_embs, (torch.cat(anc_hs, 1),
                                                                 torch.cat(anc_cs, 1)))
                else:
                    hid, (hc, cc) = self.seg_rnns[rul_ss](embs, (torch.cat(anc_hs, 1),
                                                                 torch.cat(anc_cs, 1)))
                thid, (thc, tcc) = self.ar_rnn(ar_embs, (torch.cat(anc_ths, 1),
                                                             torch.cat(anc_tcs, 1)))

            # retrieve topk for this segment (in reverse order)
            seghyps = [heapq.heappop(minq) for _ in xrange(len(minq))]
            if len(seghyps) == 0:
                return -float("inf"), None

            if len(seghyps) < K and not final_state:
                # haaaaaaaaaaaaaaack
                ugh = []
                for ick in xrange(K-len(seghyps)):
                    scoreick, lenick, hypick, thcick, tccick = seghyps[0]
                    ugh.append((scoreick - 9999999.0 + ick, lenick, hypick, thcick, tccick))
                    # break ties for the comparison
                ugh.extend(seghyps)
                seghyps = ugh

            #assert final_state or len(seghyps) == K

            if final_state:
                if len(seghyps) > 0:
                    scoreb, lenb, hypb, thcb, tccb = seghyps[-1]
                    return scoreb, backtrace3(hypb)
                else:
                    return -float("inf"), None
            else:
                thidlst, thclst, tcclst = [], [], []
                for i in xrange(K):
                    scorei, leni, hypi, thci, tcci = seghyps[K-i-1]
                    curr_scores[i][0], curr_lens[i][0], curr_hyps[i] = scorei, leni, hypi
                    thidlst.append(thci[-1:, :, :]) # each is 1 x 1 x rnn_size
                    thclst.append(thci) # each is layers x 1 x rnn_size
                    tcclst.append(tcci) # each is layers x 1 x rnn_size

                # we already have the state for the next word b/c we put it thru to also predict eop
                thid, (thc, tcc) = torch.cat(thidlst, 1), (torch.cat(thclst, 1), torch.cat(tcclst, 1))


    def gen_one_ar(self, templt, h0, c0, srcfieldenc, len_lps, row2tblent, row2feats):
        """
        src - 1 x nfields x nfeatures
        h0 - rnn_size vector
        c0 - rnn_size vector
        srcfieldenc - 1 x nfields x dim
        len_lps - K x L, log normalized
        returns a list of phrases
        """
        nout_wrds = self.decoder.out_features
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        phrases, phrs = [], []
        tokes = 0.0
        wscore, hyp = self.temp_ar_bs(templt, row2tblent, row2feats, h0, c0, srcfieldenc, len_lps,
                                   args.beamsz, corpus)
        if hyp is None:
            return None, -float("inf"), 0
        curr_labe = hyp[0][1]
        tokes = 0
        for widx, labe in hyp:
            if labe != curr_labe:
                phrases.append(phrs)
                tokes += len(phrs)
                phrs = []
                curr_labe = labe
            if widx < nout_wrds:
                phrs.append(i2w[widx])
            else:
                tblidx = widx - nout_wrds
                _, _, wordstr = row2tblent[tblidx]
                if args.verbose:
                    phrs.append(wordstr + " (c)")
                else:
                    phrs.append(wordstr)
        if len(phrs) > 0:
            phrases.append(phrs)
            tokes += len(phrs)

        return phrases, wscore, tokes

def make_combo_targs(locs, x, L, nfields, ngen_types):
    """
    combines word and copy targets into a single tensor.
    locs - seqlen x bsz x max_locs
    x - seqlen x bsz
    assumes we have word indices, then fields, then a dummy
    returns L x bsz*seqlen x max_locs tensor corresponding to xsegs[1:]
    """
    seqlen, bsz, max_locs = locs.size()
    # first replace -1s in first loc with target words
    addloc = locs + (ngen_types+1) # seqlen x bsz x max_locs
    firstloc = addloc[:, :, 0] # seqlen x bsz
    targmask = (firstloc == ngen_types) # -1 will now have value ngentypes
    firstloc[targmask] = x[targmask]
    # now replace remaining -1s w/ zero location
    addloc[addloc == ngen_types] = ngen_types+1+nfields # last index
    # finally put in same format as x_segs
    newlocs = torch.LongTensor(L, seqlen, bsz, max_locs).fill_(ngen_types+1+nfields)
    for i in xrange(L):
        newlocs[i][:seqlen-i].copy_(addloc[i:])
    return newlocs.transpose(1, 2).contiguous().view(L, bsz*seqlen, max_locs)


def get_uniq_fields(src, pad_idx, keycol=0):
    """
    src - bsz x nfields x nfeats
    """
    bsz = src.size(0)
    # get unique keys for each example
    keys = [torch.LongTensor(list(set(src[b, :, keycol]))) for b in xrange(bsz)]
    maxkeys = max(keyset.size(0) for keyset in keys)
    fields = torch.LongTensor(bsz, maxkeys).fill_(pad_idx)
    for b, keyset in enumerate(keys):
        fields[b][:len(keyset)].copy_(keyset)
    return fields


def make_masks(src, pad_idx, max_pool=False):
    """
    src - bsz x nfields x nfeats
    """
    neginf = -1e38
    bsz, nfields, nfeats = src.size()
    fieldmask = (src.eq(pad_idx).sum(2) == nfeats) # binary bsz x nfields tensor
    avgmask = (1 - fieldmask).float() # 1s where not padding
    if not max_pool:
        avgmask.div_(avgmask.sum(1, True).expand(bsz, nfields))
    fieldmask = fieldmask.float() * neginf # 0 where not all pad and -1e38 elsewhere
    return fieldmask, avgmask

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default='', help='path to data dir')
parser.add_argument('-epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=16, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200,
                    help='minibatches to wait before logging training status')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-load', type=str, default='', help='path to saved model')
parser.add_argument('-test', action='store_true', help='use test data')
parser.add_argument('-thresh', type=int, default=9, help='prune if occurs <= thresh')
parser.add_argument('-max_mbs_per_epoch', type=int, default=35000, help='max minibatches per epoch')

parser.add_argument('-emb_size', type=int, default=100, help='size of word embeddings')
parser.add_argument('-hid_size', type=int, default=100, help='size of rnn hidden state')
parser.add_argument('-layers', type=int, default=1, help='num rnn layers')
parser.add_argument('-A_dim', type=int, default=64,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('-cond_A_dim', type=int, default=32,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('-smaller_cond_dim', type=int, default=64,
                    help='dim of thing we feed into linear to get transitions')
parser.add_argument('-yes_self_trans', action='store_true', help='')
parser.add_argument('-mlpinp', action='store_true', help='')
parser.add_argument('-mlp_sz_mult', type=int, default=2, help='mlp hidsz is this x emb_size')
parser.add_argument('-max_pool', action='store_true', help='for word-fields')

parser.add_argument('-constr_tr_epochs', type=int, default=100, help='')
parser.add_argument('-no_ar_epochs', type=int, default=100, help='')

parser.add_argument('-word_ar', action='store_true', help='')
parser.add_argument('-ar_after_decay', action='store_true', help='')
parser.add_argument('-no_ar_for_vit', action='store_true', help='')
parser.add_argument('-fine_tune', action='store_true', help='only train ar rnn')

parser.add_argument('-dropout', type=float, default=0.3, help='dropout')
parser.add_argument('-emb_drop', action='store_true', help='dropout on embeddings')
parser.add_argument('-lse_obj', action='store_true', help='')
parser.add_argument('-sep_attn', action='store_true', help='')
parser.add_argument('-max_seqlen', type=int, default=70, help='')

parser.add_argument('-K', type=int, default=10, help='number of states')
parser.add_argument('-Kmul', type=int, default=1, help='number of states multiplier')
parser.add_argument('-L', type=int, default=10, help='max segment length')
parser.add_argument('-unif_lenps', action='store_true', help='')
parser.add_argument('-one_rnn', action='store_true', help='')

parser.add_argument('-initrange', type=float, default=0.1, help='uniform init interval')
parser.add_argument('-lr', type=float, default=1.0, help='initial learning rate')
parser.add_argument('-lr_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('-optim', type=str, default="sgd", help='optimization algorithm')
parser.add_argument('-onmt_decay', action='store_true', help='')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-interactive', action='store_true', help='')
parser.add_argument('-label_train', action='store_true', help='')
parser.add_argument('-gen_from_fi', type=str, default='', help='')
parser.add_argument('-verbose', action='store_true', help='')
parser.add_argument('-prev_loss', type=float, default=None, help='')
parser.add_argument('-best_loss', type=float, default=None, help='')

parser.add_argument('-tagged_fi', type=str, default='', help='path to tagged fi')
parser.add_argument('-ntemplates', type=int, default=200, help='num templates for gen')
parser.add_argument('-beamsz', type=int, default=1, help='')
parser.add_argument('-gen_wts', type=str, default='1,1', help='')
parser.add_argument('-min_gen_tokes', type=int, default=0, help='')
parser.add_argument('-min_gen_states', type=int, default=0, help='')
parser.add_argument('-gen_on_valid', action='store_true', help='')
parser.add_argument('-align', action='store_true', help='')
parser.add_argument('-wid_workers', type=str, default='', help='')

if __name__ == "__main__":
    args = parser.parse_args()
    print args

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print "WARNING: You have a CUDA device, so you should probably run with -cuda"
        else:
            torch.cuda.manual_seed(args.seed)

    # Load data
    corpus = labeled_data.SentenceCorpus(args.data, args.bsz, thresh=args.thresh, add_bos=False,
                                     add_eos=False, test=args.test)

    if not args.interactive and not args.label_train and len(args.gen_from_fi) == 0:
        # make constraint things from labels
        train_cidxs, train_fwd_cidxs = [], []
        for i in xrange(len(corpus.train)):
            x, constrs, _, _, _ = corpus.train[i]
            train_cidxs.append(make_bwd_constr_idxs(args.L, x.size(0), constrs))
            train_fwd_cidxs.append(make_fwd_constr_idxs(args.L, x.size(0), constrs))

    saved_args, saved_state = None, None
    if len(args.load) > 0:
        saved_stuff = torch.load(args.load)
        saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
        for k, v in args.__dict__.iteritems():
            if k not in saved_args.__dict__:
                saved_args.__dict__[k] = v
        net = HSMM(len(corpus.dictionary), corpus.ngen_types, saved_args)
        # for some reason selfmask breaks load_state
        del saved_state["selfmask"]
        net.load_state_dict(saved_state, strict=False)
        args.pad_idx = corpus.dictionary.word2idx["<pad>"]
        if args.fine_tune:
            for name, param in net.named_parameters():
                if name in saved_state:
                    param.requires_grad = False

    else:
        args.pad_idx = corpus.dictionary.word2idx["<pad>"]
        net = HSMM(len(corpus.dictionary), corpus.ngen_types, args)

    if args.cuda:
        net = net.cuda()

    if args.optim == "adagrad":
        optalg = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        for group in optalg.param_groups:
            for p in group['params']:
                optalg.state[p]['sum'].fill_(0.1)
    elif args.optim == "rmsprop":
        optalg = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optim == "adam":
        optalg = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    else:
        optalg = None

    def train(epoch):
        # Turn on training mode which enables dropout.
        net.train()
        neglogev = 0.0 #  negative log evidence
        nsents = 0
        trainperm = torch.randperm(len(corpus.train))
        nmini_batches = min(len(corpus.train), args.max_mbs_per_epoch)
        for batch_idx in xrange(nmini_batches):
            net.zero_grad()
            x, _, src, locs, inps = corpus.train[trainperm[batch_idx]]
            cidxs = train_cidxs[trainperm[batch_idx]] if epoch <= args.constr_tr_epochs else None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen < args.L or seqlen > args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, args.L, nfields, corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, args.pad_idx, max_pool=args.max_pool)

            uniqfields = get_uniq_fields(src, args.pad_idx) # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.cuda()
                if cidxs is not None:
                    cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            srcenc, srcfieldenc, uniqenc = net.encode(Variable(src), Variable(amask), # bsz x hid
                                                      Variable(uniqfields))
            init_logps, trans_logps = net.trans_logprobs(uniqenc, seqlen) # bsz x K, T-1 x bsz x KxK
            len_logprobs, _ = net.len_logprobs()
            fwd_obs_logps = net.obs_logprobs(Variable(inps), srcenc, srcfieldenc, Variable(fmask),
                                             Variable(combotargs), bsz) # L x T x bsz x K
            # get T+1 x bsz x K beta quantities
            beta, beta_star = infc.just_bwd(trans_logps, fwd_obs_logps,
                                            len_logprobs, constraints=cidxs)
            log_marg = logsumexp1(beta_star[0] + init_logps).sum() # bsz x 1 -> 1
            neglogev -= log_marg.data[0]
            lossvar = -log_marg/bsz
            lossvar.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
            if optalg is not None:
                optalg.step()
            else:
                for p in net.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.lr, p.grad.data)

            nsents += bsz

            if (batch_idx+1) % args.log_interval == 0:
                print "batch %d/%d | train neglogev %g " % (batch_idx+1,
                                                            nmini_batches,
                                                            neglogev/nsents)
        print "epoch %d | train neglogev %g " % (epoch, neglogev/nsents)
        return neglogev/nsents

    def test(epoch):
        net.eval()
        neglogev = 0.0
        nsents = 0

        for i in xrange(len(corpus.valid)):
            x, _, src, locs, inps = corpus.valid[i]
            cidxs = None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen < args.L or seqlen > args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, args.L, nfields, corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, args.pad_idx, max_pool=args.max_pool)

            uniqfields = get_uniq_fields(src, args.pad_idx) # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.cuda()
                if cidxs is not None:
                    cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            srcenc, srcfieldenc, uniqenc = net.encode(Variable(src, volatile=True),  # bsz x hid
                                                      Variable(amask, volatile=True),
                                                      Variable(uniqfields, volatile=True))
            init_logps, trans_logps = net.trans_logprobs(uniqenc, seqlen) # bsz x K, T-1 x bsz x KxK
            len_logprobs, _ = net.len_logprobs()
            fwd_obs_logps = net.obs_logprobs(Variable(inps, volatile=True), srcenc,
                                             srcfieldenc, Variable(fmask, volatile=True),
                                             Variable(combotargs, volatile=True),
                                             bsz) # L x T x bsz x K

            # get T+1 x bsz x K beta quantities
            beta, beta_star = infc.just_bwd(trans_logps, fwd_obs_logps,
                                            len_logprobs, constraints=cidxs)
            log_marg = logsumexp1(beta_star[0] + init_logps).sum() # bsz x 1 -> 1
            neglogev -= log_marg.data[0]
            nsents += bsz
        print "epoch %d | valid ev %g" % (epoch, neglogev/nsents)
        return neglogev/nsents

    def label_train():
        net.ar = saved_args.ar_after_decay and not args.no_ar_for_vit
        print "btw, net.ar:", net.ar
        for i in xrange(len(corpus.train)):
            x, _, src, locs, inps = corpus.train[i]
            fwd_cidxs = None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen <= saved_args.L: #or seqlen > args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, saved_args.L, nfields, corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, saved_args.pad_idx, max_pool=saved_args.max_pool)
            uniqfields = get_uniq_fields(src, args.pad_idx) # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.cuda()
                if fwd_cidxs is not None:
                    fwd_cidxs = [tens.cuda() if tens is not None else None for tens in fwd_cidxs]
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            srcenc, srcfieldenc, uniqenc = net.encode(Variable(src, volatile=True), # bsz x hid
                                                      Variable(amask, volatile=True),
                                                      Variable(uniqfields, volatile=True))
            init_logps, trans_logps = net.trans_logprobs(uniqenc, seqlen) # bsz x K, T-1 x bsz x KxK
            len_logprobs, _ = net.len_logprobs()
            fwd_obs_logps = net.obs_logprobs(Variable(inps, volatile=True), srcenc,
                                             srcfieldenc, Variable(fmask, volatile=True),
                                             Variable(combotargs, volatile=True), bsz) # LxTxbsz x K
            bwd_obs_logprobs = infc.bwd_from_fwd_obs_logprobs(fwd_obs_logps.data)
            seqs = infc.viterbi(init_logps.data, trans_logps.data, bwd_obs_logprobs,
                                [t.data for t in len_logprobs], constraints=fwd_cidxs)
            for b in xrange(bsz):
                words = [corpus.dictionary.idx2word[w] for w in x[:, b]]
                for (start, end, label) in seqs[b]:
                    print "%s|%d" % (" ".join(words[start:end]), label),
                print

    def gen_from_srctbl(src_tbl, top_temps, coeffs, src_line=None):
        net.ar = saved_args.ar_after_decay
        #print "btw2", net.ar
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        best_score, best_phrases, best_templt = -float("inf"), None, None
        best_len = 0
        best_tscore, best_gscore = None, None

        # get srcrow 2 key, idx
        #src_b = src.narrow(0, b, 1) # 1 x nfields x nfeats
        src_b = corpus.featurize_tbl(src_tbl).unsqueeze(0) # 1 x nfields x nfeats
        uniq_b = get_uniq_fields(src_b, saved_args.pad_idx) # 1 x max_fields
        if args.cuda:
            src_b = src_b.cuda()
            uniq_b = uniq_b.cuda()

        srcenc, srcfieldenc, uniqenc = net.encode(Variable(src_b, volatile=True), None,
                                                  Variable(uniq_b, volatile=True))
        init_logps, trans_logps = net.trans_logprobs(uniqenc, 2)
        _, len_scores = net.len_logprobs()
        len_lps = net.lsm(len_scores).data
        init_logps, trans_logps = init_logps.data.cpu(), trans_logps.data[0].cpu()
        inits = net.h0_lin(srcenc)
        h0, c0 = F.tanh(inits[:, :inits.size(1)/2]), inits[:, inits.size(1)/2:]

        nfields = src_b.size(1)
        row2tblent = {}
        for ff in xrange(nfields):
            field, idx = i2w[src_b[0][ff][0]], i2w[src_b[0][ff][1]]
            if (field, idx) in src_tbl:
                row2tblent[ff] = (field, idx, src_tbl[field, idx])
            else:
                row2tblent[ff] = (None, None, None)

        # get row to input feats
        row2feats = {}
        # precompute wrd stuff
        fld_cntr = Counter([key for key, _ in src_tbl])
        for row, (k, idx, wrd) in row2tblent.iteritems():
            if k in w2i:
                widx = w2i[wrd] if wrd in w2i else w2i["<unk>"]
                keyidx = w2i[k] if k in w2i else w2i["<unk>"]
                idxidx = w2i[idx]
                cheatfeat = w2i["<stop>"] if fld_cntr[k] == idx else w2i["<go>"]
                #row2feats[row] = torch.LongTensor([keyidx, idxidx, cheatfeat])
                row2feats[row] = torch.LongTensor([widx, keyidx, idxidx, cheatfeat])

        constr_sat = False
        # search over all templates
        for templt in top_temps:
            #print "templt is", templt
            # get templt transition prob
            tscores = [init_logps[0][templt[0]]]
            [tscores.append(trans_logps[0][templt[tt-1]][templt[tt]])
             for tt in xrange(1, len(templt))]

            if net.ar:
                phrases, wscore, tokes = net.gen_one_ar(templt, h0[0], c0[0], srcfieldenc,
                    len_lps, row2tblent, row2feats)
                rul_tokes = tokes
            else:
                phrases, wscore, lscore, tokes, segs = net.gen_one(templt, h0[0], c0[0],
                    srcfieldenc, len_lps, row2tblent, row2feats)
                rul_tokes = tokes - segs # subtract imaginary toke for each <eop>
                wscore /= tokes
            segs = len(templt)
            if (rul_tokes < args.min_gen_tokes or segs < args.min_gen_states) and constr_sat:
                continue
            if rul_tokes >= args.min_gen_tokes and segs >= args.min_gen_states:
                constr_sat = True # satisfied our constraint
            tscore = sum(tscores[:int(segs)])/segs
            if not net.unif_lenps:
                tscore += lscore/segs

            gscore = wscore
            ascore = coeffs[0]*tscore + coeffs[1]*gscore
            if (constr_sat and ascore > best_score) or (not constr_sat and rul_tokes > best_len) or (not constr_sat and rul_tokes == best_len and ascore > best_score):
            # take if improves score or not long enough yet and this is longer...
            #if ascore > best_score: #or (not constr_sat and rul_tokes > best_len):
                best_score, best_tscore, best_gscore = ascore, tscore, gscore
                best_phrases, best_templt = phrases, templt
                best_len = rul_tokes
            #str_phrases = [" ".join(phrs) for phrs in phrases]
            #tmpltd = ["%s|%d" % (phrs, templt[k]) for k, phrs in enumerate(str_phrases)]
            #statstr = "a=%.2f t=%.2f g=%.2f" % (ascore, tscore, gscore)
            #print "%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd)), statstr
            #assert False
        #assert False

        try:
            str_phrases = [" ".join(phrs) for phrs in best_phrases]
        except TypeError:
            # sometimes it puts an actual number in
            str_phrases = [" ".join([str(n) if type(n) is int else n for n in phrs]) for phrs in best_phrases]
        tmpltd = ["%s|%d" % (phrs, best_templt[kk]) for kk, phrs in enumerate(str_phrases)]
        if args.verbose:
            print src_line
            #print src_tbl

        print "%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd))
        if args.verbose:
            statstr = "a=%.2f t=%.2f g=%.2f" % (best_score, best_tscore, best_gscore)
            print statstr
            print
        #assert False

    def gen_from_src():
        from template_extraction import extract_from_tagged_data, align_cntr
        top_temps, _, _ = extract_from_tagged_data(args.data, args.bsz, args.tagged_fi,
                                                         args.ntemplates)

        with open(args.gen_from_fi) as f:
            src_lines = f.readlines()

        if len(args.wid_workers) > 0:
            wid, nworkers = [int(n.strip()) for n in args.wid_workers.split(',')]
            chunksz = math.floor(len(src_lines)/float(nworkers))
            startln = int(wid*chunksz)
            endln = int((wid+1)*chunksz) if wid < nworkers-1 else len(src_lines)
            print >> sys.stderr, "worker", wid, "doing lines", startln, "thru", endln-1
            src_lines = src_lines[startln:endln]

        net.eval()
        coeffs = [float(flt.strip()) for flt in args.gen_wts.split(',')]
        if args.gen_on_valid:
            for i in xrange(len(corpus.valid)):
                if i > 2:
                    break
                x, _, src, locs, inps = corpus.valid[i]
                seqlen, bsz = x.size()
                #nfields = src.size(1)
                # get bsz x nfields, bsz x nfields masks
                #fmask, amask = make_masks(src, saved_args.pad_idx, max_pool=saved_args.max_pool)
                #if args.cuda:
                    #src = src.cuda()
                    #amask = amask.cuda()

                for b in xrange(bsz):
                    src_line = src_lines[corpus.val_mb2linenos[i][b]]
                    if "wiki" in args.data:
                        src_tbl = get_wikibio_poswrds(src_line.strip().split())
                    else:
                        src_tbl = get_e2e_poswrds(src_line.strip().split())

                    gen_from_srctbl(src_tbl, top_temps, coeffs, src_line=src_line)
        else:
            for ll, src_line in enumerate(src_lines):
                if "wiki" in args.data:
                    src_tbl = get_wikibio_poswrds(src_line.strip().split())
                else:
                    src_tbl = get_e2e_poswrds(src_line.strip().split())

                gen_from_srctbl(src_tbl, top_temps, coeffs, src_line=src_line)


    def align_stuff():
        from template_extraction import extract_from_tagged_data
        i2w = corpus.dictionary.idx2word
        net.eval()
        cop_counters = [Counter() for _ in xrange(net.K*net.Kmul)]
        net.ar = saved_args.ar_after_decay and not args.no_ar_for_vit
        top_temps, _, _ = extract_from_tagged_data(args.data, args.bsz, args.tagged_fi,
                                                   args.ntemplates)
        top_temps = set(temp for temp in top_temps)

        with open(os.path.join(args.data, "train.txt")) as f:
            tgtlines = [line.strip().split() for line in f]

        with open(os.path.join(args.data, "src_train.txt")) as f:
            srclines = [line.strip().split() for line in f]

        assert len(srclines) == len(tgtlines)

        for i in xrange(len(corpus.train)):
            x, _, src, locs, inps = corpus.train[i]
            fwd_cidxs = None

            seqlen, bsz = x.size()
            nfields = src.size(1)
            if seqlen <= saved_args.L or seqlen > args.max_seqlen:
                continue

            combotargs = make_combo_targs(locs, x, saved_args.L, nfields, corpus.ngen_types)
            # get bsz x nfields, bsz x nfields masks
            fmask, amask = make_masks(src, saved_args.pad_idx, max_pool=saved_args.max_pool)
            uniqfields = get_uniq_fields(src, args.pad_idx) # bsz x max_fields

            if args.cuda:
                combotargs = combotargs.cuda()
                src = src.cuda()
                inps = inps.cuda()
                fmask, amask = fmask.cuda(), amask.cuda()
                uniqfields = uniqfields.cuda()

            srcenc, srcfieldenc, uniqenc = net.encode(Variable(src, volatile=True), # bsz x hid
                                                      Variable(amask, volatile=True),
                                                      Variable(uniqfields, volatile=True))
            init_logps, trans_logps = net.trans_logprobs(uniqenc, seqlen) # bsz x K, T-1 x bsz x KxK
            len_logprobs, _ = net.len_logprobs()
            fwd_obs_logps = net.obs_logprobs(Variable(inps, volatile=True), srcenc,
                                             srcfieldenc, Variable(fmask, volatile=True),
                                             Variable(combotargs, volatile=True), bsz) # LxTxbsz x K
            bwd_obs_logprobs = infc.bwd_from_fwd_obs_logprobs(fwd_obs_logps.data)
            seqs = infc.viterbi(init_logps.data, trans_logps.data, bwd_obs_logprobs,
                                [t.data for t in len_logprobs], constraints=fwd_cidxs)
            # get rid of stuff not in our top_temps
            for bidx in xrange(bsz):
                if tuple(labe for (start, end, labe) in seqs[bidx]) in top_temps:
                    lineno = corpus.train_mb2linenos[i][bidx]
                    tgttokes = tgtlines[lineno]
                    if "wiki" in args.data:
                        src_tbl = get_wikibio_poswrds(srclines[lineno])
                    else:
                        src_tbl = get_e2e_poswrds(srclines[lineno]) # field, idx -> wrd
                    wrd2fields = defaultdict(list)
                    for (field, idx), wrd in src_tbl.iteritems():
                        wrd2fields[wrd].append(field)
                    for (start, end, labe) in seqs[bidx]:
                        for wrd in tgttokes[start:end]:
                            if wrd in wrd2fields:
                                cop_counters[labe].update(wrd2fields[wrd])
        return cop_counters

    if args.interactive:
        pass
    elif args.align:
        cop_counters = align_stuff()
    elif args.label_train:
        net.eval()
        label_train()
    elif len(args.gen_from_fi) > 0:
        gen_from_src()
    elif args.epochs == 0:
        net.eval()
        test(0)
    else:
        prev_valloss, best_valloss = float("inf"), float("inf")
        decayed = False
        if args.prev_loss is not None:
            prev_valloss = args.prev_loss
            if args.best_loss is None:
                best_valloss = prev_valloss
            else:
                decayed = True
                best_valloss = args.best_loss
            print "starting with", prev_valloss, best_valloss

        for epoch in range(1, args.epochs + 1):
            if epoch > args.no_ar_epochs and not net.ar and decayed:
                net.ar = True
                # hack
                if args.word_ar and not net.word_ar:
                    print "turning on word ar..."
                    net.word_ar = True

            print "ar:", net.ar

            train(epoch)
            net.eval()
            valloss = test(epoch)

            if valloss < best_valloss:
                best_valloss = valloss
                if len(args.save) > 0:
                    print "saving to", args.save
                    state = {"opt": args, "state_dict": net.state_dict(),
                             "lr": args.lr, "dict": corpus.dictionary}
                    torch.save(state, args.save + "." + str(int(decayed)))

            if (args.optim == "sgd" and valloss >= prev_valloss) or (args.onmt_decay and decayed):
                decayed = True
                args.lr *= args.lr_decay
                if args.ar_after_decay and not net.ar:
                    net.ar = True
                    # hack
                    if args.word_ar and not net.word_ar:
                        print "turning on word ar..."
                        net.word_ar = True
                print "decaying lr to:", args.lr
                if args.lr < 1e-5:
                    break
            prev_valloss = valloss
            if args.cuda:
                print "ugh...."
                shmocals = locals()
                for shk in shmocals.keys():
                    shv = shmocals[shk]
                    if hasattr(shv, "is_cuda") and shv.is_cuda:
                        shv = shv.cpu()
                print "done!"
                print
            else:
                print
