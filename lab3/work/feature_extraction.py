import os
from os.path import join
from os import walk
from tqdm import tqdm
import numpy as np
from lab3_tools import loadAudio
from lab3_tools import path2info
from lab1_proto import mspec
from lab1_proto import mfcc
from prondict import prondict
from lab3_proto import words2phones
from lab3_proto import forcedAlignment

if __name__ == "__main__":
    # config
    folder_to_extract = "tidigits/disc_4.1.1/tidigits/train"
    dump_file_name = "traindata.npz"
    add_short_pause = True
    data_type = 'traindata'
    # ==================================================
    # preparing
    phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
    # get the number of states of each phoen
    nstates = dict()
    for ph in phoneHMMs.keys():
        num_state = phoneHMMs[ph]['means'].shape[0]
        nstates[ph] = num_state
    # create a state list
    # split a state by the number of states in its HMM.
    # E.g. # of states of ah = 3; ah -> ['ah_0', 'ah_1', 'ah_2']
    stateList = list()
    for ph in phoneHMMs.keys():
        for i in range(nstates[ph]):
            stateList.append('%s_%d' % (ph, i))
    # --------------------------------------------------------------
    data = list()
    for root, dirs, files in walk(folder_to_extract):
        for f in tqdm(files):
            if not f.endswith('.wav'):
                continue
            # do our work
            filename = os.path.join(root, f)
            sample, srate = loadAudio(filename)
            mspec_x = mspec(sample, samplingrate=srate)
            lmfcc_x = mfcc(sample, samplingrate=srate)
            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict)
            targets = forcedAlignment(lmfcc_x, phoneHMMs, phoneTrans)
            # convert the targets from str to int
            idx_targets = [stateList.index(t) for t in targets]
            data.append(
                {
                    'filename': filename,
                    'lmfcc': lmfcc_x,
                    'mspec': mspec_x,
                    'targets': idx_targets
                }
            )

    kwargs = {
        data_type: data
    }
    np.savez(dump_file_name, **kwargs)
