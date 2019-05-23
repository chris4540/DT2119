"""
For section 4.4 Part 1

Script to split the full training to validation set and training set

Get the list of speaker in validation set and save it to the json file:
val_spk.json
"""
import numpy as np
import json
from lab3_tools import path2info

# set seed
np.random.seed(10)

if __name__ == "__main__":
    # config
    num_speaker = 77
    num_man = 55  # count them by `ls -1 | wc -l`
    num_woman = 57
    valid_part = 0.1  # ~10%
    # ============================================
    val_n_man = int(num_man * valid_part)
    val_n_woman = int(num_woman * valid_part)
    val_spk_id = set()
    val_data = list()
    trian_data = list()
    print("# of validation of woman speakers", val_n_woman)
    print("# of validation of man speakers", val_n_man)
    # load the train data
    traindata = np.load('data/traindata.npz')['traindata']
    # loop over data set
    for d in traindata:
        sound_fname = d['filename']
        gender, speakerID, _, _ = path2info(sound_fname)
        spk_gen_id = '{}-{}'.format(gender, speakerID)

        if gender == 'man' and val_n_man > 0 and spk_gen_id not in val_spk_id:
            opt = np.random.choice(['val', 'train'], size=1)[0]
            if opt == 'val':
                val_spk_id.add(spk_gen_id)
                val_n_man -= 1
        elif gender == 'woman' and val_n_woman > 0 and spk_gen_id not in val_spk_id:
                val_spk_id.add(spk_gen_id)
                val_n_woman -= 1

    with open('val_spk.json', 'w') as f:
        data = {
            'valdation_spker': list(val_spk_id)
        }
        json.dump(data, f)