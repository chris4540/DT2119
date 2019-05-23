"""
Data normalization for 4.6
"""
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # feature_name = 'lmfcc'
    feature_name = 'mspec'

    print("Working on the feature: ", feature_name)
    # load data
    data = np.load('data/train_val_data.npz')
    valdata = data['validation']
    traindata = data['train']
    testdata = np.load('data/testdata.npz')['testdata']

    # Calculate the mean and std first
    print("[Non-dynamic feature] Calculate the mean and std ....")
    # for each utterance generate dynamic features
    scaler = StandardScaler()

    for d in tqdm(traindata):
        feature_mat = d[feature_name]
        scaler.partial_fit(feature_mat) # do online fitting
    print("[Non-dynamic feature] Complete calculate the standard scaler")
    # =========================================================
    datasets = {
        'train': traindata,
        'test': testdata,
        'val': valdata,
    }

    for k, v in datasets.items():
        data = list()
        for d in tqdm(v):
            new_data = dict()
            # new_data['filename'] = d['filename']
            new_data['targets'] = d['targets']
            # use half precision
            new_data[feature_name] = scaler.transform(d[feature_name]).astype("float16")
            data.append(new_data)

        print("[Non-dynamic feature]Complete normlaizating ", k)
        # save it
        np.savez('data/nondyn/{}_{}.npz'.format(feature_name, k), data=data)
        print("[Non-dynamic feature] Wrote the normlaized data", k)

