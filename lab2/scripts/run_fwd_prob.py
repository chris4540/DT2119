"""
For the part 5.2 Forward Algorithm

Run this script
    %run scripts/run_fwd_prob.py
"""
import numpy as np
import pandas as pd
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import forward
from lab2_tools import logsumexp
from lab2_tools import log_multivariate_normal_density_diag

def get_loglik(feature, hmm):
    trans_mat = hmm['transmat'][:-1,:-1]
    pi_vec = hmm['startprob'][:-1]
    means = hmm['means']
    covars = hmm['covars']
    obsloglik = log_multivariate_normal_density_diag(feature, means, covars)
    log_alpha = forward(obsloglik, np.log(pi_vec), np.log(trans_mat))
    ret = logsumexp(log_alpha[-1])
    return ret

def match_model_and_utterances(dataset, wordHMMs):
    ret = list()
    for data in dataset:
        fields = [data[k] for k in ['digit', 'gender', 'speaker', 'repetition']]
        label = "_".join(fields)
        lmfcc = data['lmfcc']
        max_loglik = -np.inf
        for d in wordHMMs.keys():
            loglik = get_loglik(lmfcc, wordHMMs[d])
            if loglik > max_loglik:
                max_loglik = loglik
                matched_model = d

        ret.append({
            'digit': data['digit'],
            'label': label,
            'matched_model': matched_model,
            'score': max_loglik
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
    log_alpha = forward(example['obsloglik'], np.log(pi_vec), np.log(trans_mat))

    # calculate the sequence log likelihood to this hmm model
    # i.e. log[P(X_{1:T} | HMM model)]
    log_seq_likelihood = logsumexp(log_alpha[-1])
    print("The log likelihood of the observation seq to the model:", log_seq_likelihood)
    # ==================================================================================
    # check if closed to the example
    # is_alpha_close = np.allclose(log_alpha, example['logalpha'])
    # print("Is closed to the example['logalpha']:", is_alpha_close)
    # score all the 44 utterances in the data array with each of the 11 HMM models in wordHMMs.
    for d in isolated.keys():
        wordHMMs[d] = concatHMMs(phoneHMMs, isolated[d])

    resp = match_model_and_utterances(data, wordHMMs)
    onespkr_match = pd.DataFrame(resp)

    phoneHMMs_all = np.load('data/lab2_models_all.npz')['phoneHMMs'].item()
    for d in isolated.keys():
        wordHMMs[d] = concatHMMs(phoneHMMs_all, isolated[d])

    resp = match_model_and_utterances(data, wordHMMs)
    all_match = pd.DataFrame(resp)