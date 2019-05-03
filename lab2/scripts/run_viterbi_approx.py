"""
For the part 5.3 Viterbi Approximation
"""
import pandas as pd
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import viterbi
from lab2_tools import logsumexp
from lab2_tools import log_multivariate_normal_density_diag
from time import time

def get_best_path_loglik(feature, hmm):
    trans_mat = hmm['transmat'][:-1,:-1]
    pi_vec = hmm['startprob'][:-1]
    means = hmm['means']
    covars = hmm['covars']

    obsloglik = log_multivariate_normal_density_diag(feature, means, covars)
    ret, _ = viterbi(obsloglik, np.log(pi_vec), np.log(trans_mat))
    return ret

def match_model_and_utterances(dataset, wordHMMs):
    ret = list()
    for data in dataset:
        fields = [data[k] for k in ['digit', 'gender', 'speaker', 'repetition']]
        label = "_".join(fields)
        lmfcc = data['lmfcc']
        score = -np.inf
        for d in wordHMMs.keys():
            loglik = get_best_path_loglik(lmfcc, wordHMMs[d])
            if loglik > score:
                score = loglik
                matched_model = d

        ret.append({
            'digit': data['digit'],
            'label': label,
            'matched_model': matched_model,
            'score': score
        })
    return ret

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
    # ========================================================================
    onespkr_wordHMMs = {}
    for k in isolated.keys():
        onespkr_wordHMMs[k] = concatHMMs(phoneHMMs, isolated[k])

    phoneHMMs_all = np.load('data/lab2_models_all.npz')['phoneHMMs'].item()
    for d in isolated.keys():
        wordHMMs[d] = concatHMMs(phoneHMMs_all, isolated[d])

    st = time()
    resp = match_model_and_utterances(data, onespkr_wordHMMs)
    onespkr_match = pd.DataFrame(resp)
    resp = match_model_and_utterances(data, wordHMMs)
    all_match = pd.DataFrame(resp)
    print("Time used for comparing models: ", time() - st)