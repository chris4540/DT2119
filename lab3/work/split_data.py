"""
4.4  Training and Validation Sets
Split data into trianing and validation set
"""

import numpy as np
import json
from lab3_tools import path2info
from tqdm import tqdm

if __name__ == "__main__":
    # load the validation speaker list; from the result of the script:
    # work/get_val_spk_json.py
    with open('data/val_spk.json', 'r') as f:
        data = json.load(f)
        val_spk = data['valdation_spker']

     # load the train data
    traindata = np.load('data/traindata.npz')['traindata']

    train = list()
    validation = list()
    for d in tqdm(traindata):
        sound_fname = d['filename']
        gender, speakerID, _, _ = path2info(sound_fname)
        spk_gen_id = '{}-{}'.format(gender, speakerID)

        if spk_gen_id in val_spk:
            validation.append(d)
        else:
            train.append(d)
    # ===================================
    print("The size of training set : ", len(train))
    print("The size of validation set : ", len(validation))
    kwargs = {
        'validation': validation,
        'train': train
    }
    np.savez('data/train_val_data.npz', **kwargs)