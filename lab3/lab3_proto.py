import numpy as np
# from lab3_tools import *
from lab2_proto import viterbi
from lab2_tools import log_multivariate_normal_density_diag as log_mnd_diag
from lab2_proto import concatHMMs

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    ret = []
    for w in wordList:
        ret = ret + pronDict[w]
        if addShortPause:
            ret += ['sp']

    if addSilence:
        ret = ["sil"] + ret + ["sil"]
    return ret


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
        if return_syb:
            list of strings in the form phoneme_index specifying, for each time step
            the state from phoneHMMs corresponding to the viterbi path.
    """
    # Obtain the mapping from state to number of state
    nstates = dict()
    for ph in phoneHMMs.keys():
        num_state = phoneHMMs[ph]['means'].shape[0]
        nstates[ph] = num_state
    # Obtain a mapping from the phoneHMMs to statename
    stateTrans = list()
    for ph in phoneTrans:
        for i in range(nstates[ph]):
            stateTrans.append("%s_%i" % (ph, i))
    # ===========================================================
    # Create the hmm model for this utterance with only the information
    # of transcription
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # calculate the Viterbi path
    means = utteranceHMM['means']
    covars = utteranceHMM['covars']
    log_emlik = log_mnd_diag(lmfcc, means, covars)
    # get \pi and A; ignore the terminal state
    log_pi = np.log(utteranceHMM['startprob'][:-1])
    log_trans = np.log(utteranceHMM['transmat'][:-1,:-1])
    _, path = viterbi(log_emlik, log_pi, log_trans)
    # =========================================================
    ret = [stateTrans[i] for i in path]

    return ret


def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
    pass