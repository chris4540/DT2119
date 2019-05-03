"""
For the part 5.4 Backward Algorithm
TODO:
    make this to be a test case
Run this script
    ipython test/test_fwd_prob.py
    %run test/test_fwd_prob.py
"""
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import backward
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

    # check the result of log_beta
    print("Is beta close?:", np.allclose(log_beta, example['logbeta']))

    # calculate the log prob
    # seq_likelihood = sum_{i}(\alpha_0(i) * \beta_0(i))
    log_alpha_0 = log_startprob.T + log_emlik[0]
    log_beta_0 = log_beta[0, :]
    log_seq_like = logsumexp(log_alpha_0 + log_beta_0)
    print(log_seq_like)
    print(example['loglik'])