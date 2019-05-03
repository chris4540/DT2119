"""
For the part 5.2 Forward Algorithm

Run this script
    %run test/test_fwd_prob.py
"""
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import forward
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
    # =====================================================
    log_alpha = forward(example['obsloglik'], np.log(pi_vec), np.log(trans_mat))

    # calculate the sequence log likelihood to this hmm model
    # i.e. log[P(X_{1:T} | HMM model)]
    log_seq_likelihood = logsumexp(log_alpha[-1])
    # print("The log likelihood of the observation seq to the model:", log_seq_likelihood)

    # check if closed to the example
    # is_alpha_close = np.allclose(log_alpha, example['logalpha'])
    # print("Is closed to the example['logalpha']:", is_alpha_close)