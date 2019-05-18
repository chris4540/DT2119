"""
4.1 Target class def
"""
import numpy as np

if __name__ == "__main__":
    phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    print("Size of phones:", len(phones))
    # -----------------------------------
    # get the number of states of each phoen
    nstates = dict()
    for ph in phones:
        num_state = phoneHMMs[ph]['means'].shape[0]
        nstates[ph] = num_state
    # ----------------------------------------
    # create a state list
    # split a state by the number of states in its HMM.
    # E.g. # of states of ah = 3; ah -> ['ah_0', 'ah_1', 'ah_2']
    stateList = list()
    for ph in phones:
        for i in range(nstates[ph]):
            stateList.append('%s_%d' % (ph, i))
    # =================================================
    assert stateList[8] == "ay_2"