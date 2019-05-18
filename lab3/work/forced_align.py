"""
4.2 forced alignment

Test the forced alignment
"""
from lab1_proto import mfcc
from lab3_tools import loadAudio
from lab3_tools import path2info
from lab3_proto import words2phones
from prondict import prondict
from lab2_proto import concatHMMs
from lab3_tools import frames2trans
import numpy as np
# ==============
from lab3_proto import forcedAlignment

if __name__ == "__main__":
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    # Get the mfcc feature vectors
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    # =================================================
    # Get the word level transcription
    wordTrans = list(path2info(filename)[2])
    # get the phone level transcription
    phoneTrans = words2phones(wordTrans, prondict)
    # print(phoneTrans)
    # ==================================================================
    # combine the HMMs according to the phone level transcription
    phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    # translate the state (idx) to a name
    phones = sorted(phoneHMMs.keys())
    nstates = dict()
    for ph in phones:
        num_state = phoneHMMs[ph]['means'].shape[0]
        nstates[ph] = num_state
    #
    stateTrans = list()
    for ph in phoneTrans:
        for i in range(nstates[ph]):
            stateTrans.append("%s_%i" % (ph, i))
    # print(stateTrans)
    assert stateTrans[10] == 'r_1'
    # ==============================================
    symb = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

    frames2trans(symb, outfilename='z43a.lab')
