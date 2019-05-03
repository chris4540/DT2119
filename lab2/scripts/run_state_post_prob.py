"""
For the part 6.1 State posterior probabilities
"""
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import backward
from lab2_proto import forward
from lab2_proto import statePosteriors
from lab2_tools import logsumexp

if __name__ == "__main__":
    # load data
    data = np.load('data/lab2_data.npz')['data']
    example = np.load('data/lab2_example.npz')['example'].item()
    phoneHMMs = np.load('data/lab2_models_onespkr.npz')['phoneHMMs'].item()

    # Build hmm
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])

    trans_mat = wordHMMs['o']['transmat'][:-1,:-1]
    pi_vec = wordHMMs['o']['startprob'][:-1]
    log_startprob = np.log(pi_vec)
    log_transmat = np.log(trans_mat)
    log_emlik = example['obsloglik']
    # =====================================================
    log_beta = backward(log_emlik, log_startprob, log_transmat)
    log_alpha = forward(log_emlik, log_startprob, log_transmat)

    # caculate the log gamma
    log_gamma = statePosteriors(log_alpha, log_beta)

    # print(np.allclose(example['loggamma'], log_gamma))

    # check if sum to one in linear / sum to zero in log domain
    ntime_steps = log_gamma.shape[0]
    zeros = np.zeros((ntime_steps))
    # print(np.allclose(logsumexp(log_gamma, axis=1), zeros))
    # ====================================================================
    # See notes, just do the normalization to get the state posterior of GMM
    logZ = logsumexp(log_emlik, axis=1)  # the normalization factor in log space
    log_gamma_gmm = log_emlik - logZ[:, np.newaxis]
