import numpy as np
from lab2_tools import logsumexp
from lab2_tools import log_multivariate_normal_density_diag

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatTwoHMMs(phoneHMMs['sil'], phoneHMMs['ow'])
       twoHMMs = concatTwoHMMs(phoneHMMs['sp'], phoneHMMs['ow'])
       twoHMMs = concatTwoHMMs(phoneHMMs['ow'], phoneHMMs['iy'])
       twoHMMs = concatTwoHMMs(phoneHMMs['iy'], phoneHMMs['sp'])

    See also:
        the concatenating_hmms.pdf document in the lab package
    """
    # ==========================================
    # concat the initial states
    # the last state of the first hmm model
    pi_last = hmm1['startprob'][-1]
    startprob = np.hstack((hmm1['startprob'][0:-1], pi_last*hmm2['startprob']))
    # ==========================================
    # trans = trans_1   | trans1_merge_pi2
    #         ---------------------------------
    #         zeros_mat | trans_2
    # Notes: trans_1 dropped the terminal state
    # =========================================
    # the top-right block matrix in the pdf (last state of hmm1 * hmm2_pi)
    trans1_merge_pi2 = hmm1['transmat'][:-1, -1][:,np.newaxis]*hmm2['startprob']
    # drop the terminal state of the first hmm
    trans_1 = hmm1['transmat'][:-1, :-1]
    # the zero block matrix at the bottom-left corner
    zeros_mat = np.zeros((hmm2['transmat'].shape[0], trans_1.shape[1]))
    # put them together
    transmat = np.block([
                [trans_1        , trans1_merge_pi2],
                [zeros_mat      , hmm2['transmat']]])
    # =================================
    # merge means and covars
    means = np.vstack((hmm1['means'], hmm2['means']))
    covars = np.vstack((hmm1['covars'], hmm2['covars']))
    # ========================
    # build the result
    ret = {
        "startprob": startprob,
        "transmat": transmat,
        "means": means,
        "covars": covars
    }
    return ret

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name.
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    pass

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    # follow the appendix in the question pdf
    log_alpha = np.zeros(log_emlik.shape)
    log_alpha[0][:] = log_startprob.T + log_emlik[0]

    for n in range(1,len(log_alpha)): # time dimension
        for i in range(log_alpha.shape[1]): # loop over states
            log_alpha[n, i] = logsumexp(log_alpha[n - 1] + log_transmat[:,i]) + log_emlik[n,i]
    return log_alpha

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    log_beta = np.zeros(log_emlik.shape)  # since beta_N = 1 => log_beta_N = 0
    for n in reversed(range(log_emlik.shape[0] - 1)): # time dimension
        for i in range(log_emlik.shape[1]): # loop over states
            log_beta[n, i] = logsumexp(log_transmat[i,:] + log_emlik[n + 1, :] + log_beta[n + 1,:])
    return log_beta

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    B = np.zeros(log_emlik.shape, dtype=int)
    V = np.zeros(log_emlik.shape)
    V[0] = log_startprob.flatten() + log_emlik[0]

    for n in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            V[n, j] = np.max(V[n-1, :] + log_transmat[:, j]) + log_emlik[n, j]
            B[n, j] = np.argmax(V[n-1, :] + log_transmat[:, j])

    best_path_loglik = np.max(V[-1, :])
    # --------------------------------------------------------------------
    # Backtrack to take viteri path
    path = list()
    state = np.argmax(V[-1, :])
    # add to the front
    path.insert(0, state)
    for n in reversed(range(log_emlik.shape[0] - 1)): # time dimension
        state = B[n+1, state]
        path.insert(0, state)

    return best_path_loglik, np.array(path)

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    pass

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    pass