import numpy as np
from lab3_tools import *

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
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    phones = sorted(phoneHMMs.keys())
    nstates = {i:phoneHMMs[i]["means"].shape[0] for i in phones}

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                for stateid in range(nstates[phone])]


    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    logMulti = log_multivariate_normal_density_diag(Imfcc, utteranceHMM["means"], utteranceHMM["covars"])
    viterbiMatrix = viterbi(logMulti.T, np.log(utteranceHMM["startprob"]), np.log(utteranceHMM["transmat"]))
    viterbiStateTrans = [stateTrans[int(i)] for i in viterbiMatrix[1]]
    return viterbiPathState


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