"""
all the inference stuff
"""

import math
import torch
from torch.autograd import Variable
from utils import logsumexp0, logsumexp2


def recover_bps(delt, bps, bps_star):
    """
    delt, bps, bps_star - seqlen+1 x bsz x K
    returns:
       bsz-length list of lists with (start_idx, end_idx, label) entries
    """
    seqlenp1, bsz, K = delt.size()
    seqlen = seqlenp1 - 1
    seqs = []
    for b in xrange(bsz):
        seq = []
        _, last_lab = delt[seqlen][b].max(0)
        last_lab = last_lab[0]
        curr_idx = seqlen # 1-indexed
        while True:
            last_len = bps[curr_idx][b][last_lab]
            seq.append((curr_idx-last_len, curr_idx, last_lab)) # start_idx, end_idx, label, 0-idxd
            curr_idx -= last_len
            if curr_idx == 0:
                break
            last_lab = bps_star[curr_idx][b][last_lab]
        seqs.append(seq[::-1])
    return seqs


def viterbi(pi, trans_logprobs, bwd_obs_logprobs, len_logprobs, constraints=None, ret_delt=False):
    """
    pi               - 1 x K
    bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
    trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t).
    see https://hal.inria.fr/hal-01064672v2/document
    """
    neginf = -1e38
    L, seqlen, bsz, K = bwd_obs_logprobs.size()
    delt = trans_logprobs.new(seqlen+1, bsz, K).fill_(-float("inf"))
    delt_star = trans_logprobs.new(seqlen+1, bsz, K).fill_(-float("inf"))
    delt_star[0].copy_(pi.expand(bsz, K))

    # currently len_logprobs contains tensors that are [1 step back; 2 steps back; ... L steps_back]
    # but we need to flip on the 0'th axis
    flipped_len_logprobs = []
    for l in xrange(len(len_logprobs)):
        llps = len_logprobs[l]
        flipped_len_logprobs.append(torch.stack([llps[-i-1] for i in xrange(llps.size(0))]))

    bps = delt.long().fill_(L)
    bps_star = delt_star.long()
    bps_star[0].copy_(torch.arange(0, K).view(1, K).expand(bsz, K))

    mask = trans_logprobs.new(L, bsz, K)

    for t in xrange(1, seqlen+1):
        steps_back = min(L, t)
        steps_fwd = min(L, seqlen-t+1)

        if steps_back <= steps_fwd:
            # steps_fwd x K -> steps_back x K
            len_terms = flipped_len_logprobs[min(L-1, steps_fwd-1)][-steps_back:]
        else: # we need to pick probs from different distributions...
            len_terms = torch.stack([len_logprobs[min(L, seqlen+1-t+jj)-1][jj]
                                     for jj in xrange(L-1, -1, -1)])

        if constraints is not None and constraints[t] is not None:
            tmask = mask.narrow(0, 0, steps_back).zero_()
            # steps_back x bsz x K -> steps_back*bsz x K
            tmask.view(-1, K).index_fill_(0, constraints[t], neginf)

        # delt_t(j) = log \sum_l p(x_{t-l+1:t}) delt*_{t-l} p(l_t)
        delt_terms = (delt_star[t-steps_back:t] # steps_back x bsz x K
                      + bwd_obs_logprobs[-steps_back:, t-1]) # steps_back x bsz x K (0-idx)
        #delt_terms.sub_(bwd_maxlens[t-steps_back:t].expand_as(delt_terms)) # steps_back x bsz x K
        delt_terms.add_(len_terms.unsqueeze(1).expand(steps_back, bsz, K))

        if constraints is not None and constraints[t] is not None:
            delt_terms.add_(tmask)

        maxes, argmaxes = torch.max(delt_terms, 0) # 1 x bsz x K, 1 x bsz x K
        delt[t] = maxes.squeeze(0)   # bsz x K
        #bps[t] = argmaxes.squeeze(0) # bsz x K
        bps[t].sub_(argmaxes.squeeze(0)) # keep track of steps back taken: L - argmax
        if steps_back < L:
            bps[t].sub_(L - steps_back)
        if t < seqlen:
            # delt*_t(k) = log \sum_j delt_t(j) p(q_{t+1}=k | q_t = j)
            # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
            tps = trans_logprobs[t-1] # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-indexed
            delt_t = delt[t] # bsz x K, viz, p(x, j)
            delt_star_terms = (tps.transpose(0, 1) # K x bsz x K
                               + delt_t.unsqueeze(2).expand(bsz, K, K).transpose(0, 1))
            maxes, argmaxes = torch.max(delt_star_terms, 0) # 1 x bsz x K, 1 x bsz x K
            delt_star[t] = maxes.squeeze(0)
            bps_star[t] = argmaxes.squeeze(0)

    #return delt, delt_star, bps, bps_star, recover_bps(delt, bps, bps_star)
    if ret_delt:
        return recover_bps(delt, bps, bps_star), delt[-1] # bsz x K total scores
    else:
        return recover_bps(delt, bps, bps_star)


def just_fwd(pi, trans_logprobs, bwd_obs_logprobs, constraints=None):
    """
    pi               - bsz x K
    bwd_obs_logprobs - L x T x bsz x K, obs probs ending at t
    trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t)
    """
    neginf = -1e38 # -float("inf")
    L, seqlen, bsz, K = bwd_obs_logprobs.size()
    # we'll be 1-indexed for alphas and betas
    alph = [None]*(seqlen+1)
    alph_star = [None]*(seqlen+1)
    alph_star[0] = pi
    mask = trans_logprobs.new(L, bsz, K)

    bwd_maxlens = trans_logprobs.new(seqlen).fill_(L) # store max possible length generated from t
    bwd_maxlens[-L:].copy_(torch.arange(L, 0, -1))
    bwd_maxlens = bwd_maxlens.log_().view(seqlen, 1, 1)

    for t in xrange(1, seqlen+1):
        steps_back = min(L, t)

        if constraints is not None and constraints[t] is not None:
            tmask = mask.narrow(0, 0, steps_back).zero_()
            # steps_back x bsz x K -> steps_back*bsz x K
            tmask.view(-1, K).index_fill_(0, constraints[t], neginf)

        # alph_t(j) = log \sum_l p(x_{t-l+1:t}) alph*_{t-l} p(l_t)
        alph_terms = (torch.stack(alph_star[t-steps_back:t]) # steps_back x bsz x K
                      + bwd_obs_logprobs[-steps_back:, t-1] # steps_back x bsz x K (0-idx)
                      - bwd_maxlens[t-steps_back:t].expand(steps_back, bsz, K))

        if constraints is not None and constraints[t] is not None:
            alph_terms = alph_terms + tmask #Variable(tmask)

        alph[t] = logsumexp0(alph_terms) # bsz x K

        if t < seqlen:
            # alph*_t(k) = log \sum_j alph_t(j) p(q_{t+1}=k | q_t = j)
            # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
            tps = trans_logprobs[t-1] # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-indexed
            alph_t = alph[t] # bsz x K, viz, p(x, j)
            alph_star_terms = (tps.transpose(0, 1) # K x bsz x K
                               + alph_t.unsqueeze(2).expand(bsz, K, K).transpose(0, 1))
            alph_star[t] = logsumexp0(alph_star_terms)

    return alph, alph_star


def just_bwd(trans_logprobs, fwd_obs_logprobs, len_logprobs, constraints=None):
    """
    fwd_obs_logprobs - L x T x bsz x K, obs probs starting at t
    trans_logprobs   - T-1 x bsz x K x K, trans_logprobs[t] = p(q_{t+1} | q_t)
    """
    neginf = -1e38 # -float("inf")
    L, seqlen, bsz, K = fwd_obs_logprobs.size()

    # we'll be 1-indexed for alphas and betas
    beta = [None]*(seqlen+1)
    beta_star = [None]*(seqlen+1)
    beta[seqlen] = Variable(trans_logprobs.data.new(bsz, K).zero_())
    mask = trans_logprobs.data.new(L, bsz, K)

    for t in xrange(1, seqlen+1):
        steps_fwd = min(L, t)

        len_terms = len_logprobs[min(L-1, steps_fwd-1)] # steps_fwd x K

        if constraints is not None and constraints[seqlen-t+1] is not None:
            tmask = mask.narrow(0, 0, steps_fwd).zero_()
            # steps_fwd x bsz x K -> steps_fwd*bsz x K
            tmask.view(-1, K).index_fill_(0, constraints[seqlen-t+1], neginf)

        # beta*_t(k) = log \sum_l beta_{t+l}(k) p(x_{t+1:t+l}) p(l_t)
        beta_star_terms = (torch.stack(beta[seqlen-t+1:seqlen-t+1+steps_fwd]) # steps_fwd x bsz x K
                           + fwd_obs_logprobs[:steps_fwd, seqlen-t] # steps_fwd x bsz x K
                           #- math.log(steps_fwd)) # steps_fwd x bsz x K
                           + len_terms.unsqueeze(1).expand(steps_fwd, bsz, K))

        if constraints is not None and constraints[seqlen-t+1] is not None:
            beta_star_terms = beta_star_terms + Variable(tmask)

        beta_star[seqlen-t] = logsumexp0(beta_star_terms)
        if seqlen-t > 0:
            # beta_t(j) = log \sum_k beta*_t(k) p(q_{t+1} = k | q_t=j)
            betastar_nt = beta_star[seqlen-t] # bsz x K
            # get bsz x K x K trans logprobs, viz., p(q_{t+1}=j|i) w/ 0th dim i, 2nd dim j
            tps = trans_logprobs[seqlen-t-1] # N.B. trans_logprobs[t] is p(q_{t+1}) and 0-idxed
            beta_terms = betastar_nt.unsqueeze(1).expand(bsz, K, K) + tps # bsz x K x K
            beta[seqlen-t] = logsumexp2(beta_terms) # bsz x K


    return beta, beta_star


# [p0    p1    p2    p3   p4
# p0:1  p1:2  p2:3  3:4  4:5
# p0:2  p1:3  2:4   3:5  4:6 ]



# so bwd log probs look like
# -inf  -inf  p1:3  p2:4
# -inf  p1:2  p2:3  p3:4
# p1    p2    p3    p4
def bwd_from_fwd_obs_logprobs(fwd_obs_logprobs):
    """
    fwd_obs_logprobs - L x T x bsz x K,
       where fwd_obs_logprobs[:,t,:,:] gives p(x_t), p(x_{t:t+1}), ..., p(x_{t:t+l})
    returns:
      bwd_obs_logprobs - L x T x bsz x K,
        where bwd_obs_logprobs[:,t,:,:] gives p(x_{t-L+1:t}), ..., p(x_{t})
    iow, fwd_obs_logprobs gives probs of segments starting at t, and bwd_obs_logprobs
    gives probs of segments ending at t
    """
    L = fwd_obs_logprobs.size(0)
    bwd_obs_logprobs = fwd_obs_logprobs.new().resize_as_(fwd_obs_logprobs).fill_(-float("inf"))
    bwd_obs_logprobs[L-1].copy_(fwd_obs_logprobs[0])
    for l in xrange(1, L):
        bwd_obs_logprobs[L-l-1, l:].copy_(fwd_obs_logprobs[l, :-l])
    return bwd_obs_logprobs
