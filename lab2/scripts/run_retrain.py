"""
For section 6.2 Retraining the emission probability distributions
"""
import numpy as np
from prondict import isolated
from lab2_proto import concatHMMs
from lab2_proto import backward
from lab2_proto import forward
from lab2_proto import statePosteriors
from lab2_proto import updateMeanAndVar
from lab2_tools import logsumexp
from lab2_tools import log_multivariate_normal_density_diag

def retrain(feature, model, num_iters=20, threshold=1):

    # extract params
    means = model['means']
    covars = model['covars']
    transmat = model['transmat'][:-1,:-1]
    startprob = model['startprob'][:-1]
    log_pi = np.log(startprob)
    log_trans = np.log(transmat)

    # calculate the emission
    obsloglik = log_multivariate_normal_density_diag(feature, means, covars)

    # EM algorithm
    loglik_old = -np.inf
    for iter_ in range(num_iters):
        # E-step
        log_alpha = forward(obsloglik, log_pi, log_trans)
        log_beta = backward(obsloglik, log_pi, log_trans)
        log_gamma = statePosteriors(log_alpha, log_beta)

        # M-step
        means, covars = updateMeanAndVar(feature, log_gamma)
        # update
        obsloglik = log_multivariate_normal_density_diag(feature, means, covars)

        loglik = logsumexp(log_alpha[-1])
        print("Iter {}: The log likelihood in EM:".format(iter_), loglik)

        # check if terminate EM
        if (loglik - loglik_old) < threshold or np.isnan(loglik):
            print("Terminating the EM")
            break
        else:
            loglik_old = loglik


if __name__ == "__main__":
    # load data
    data = np.load('data/lab2_data.npz')['data']
    phoneHMMs = np.load('data/lab2_models_onespkr.npz')['phoneHMMs'].item()


    # Build hmm
    wordHMMs = {}
    for d in isolated.keys():
        wordHMMs[d] = concatHMMs(phoneHMMs, isolated[d])

    # get the observation sequence
    feature = data[10]['lmfcc']

    # First part
    # calculate the emissions
    digit = '4'
    means = wordHMMs[digit]['means']
    covars = wordHMMs[digit]['covars']
    obsloglik = log_multivariate_normal_density_diag(feature, means, covars)

    # calculate the log likelihood
    trans_mat = wordHMMs[digit]['transmat'][:-1,:-1]
    pi_vec = wordHMMs[digit]['startprob'][:-1]
    # log space
    log_pi = np.log(pi_vec)
    log_trans = np.log(trans_mat)

    log_alpha = forward(obsloglik, log_pi, log_trans)
    log_seq_likelihood = logsumexp(log_alpha[-1])
    print("The log likelihood of the digit {}:".format(digit), log_seq_likelihood)
    # =========================================================================
    for d in wordHMMs.keys():
        print("========================================================")
        print("Retrain HMM of digit '{}' with data[10]['lmfcc']".format(d))
        retrain(feature, wordHMMs[d])
        print("========================================================")
