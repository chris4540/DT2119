"""
For the part 5.3 Viterbi Approximation
TODO:
    make this to be a test case
Run this script
    ipython test/test_fwd_prob.py
    %run test/test_fwd_prob.py
"""
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import viterbi
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
    best_seq_loglik, best_path = viterbi(example['obsloglik'], np.log(pi_vec), np.log(trans_mat))
    assert np.allclose(best_seq_loglik, example['vloglik'])
    assert np.array_equal(best_path, example['vpath'])