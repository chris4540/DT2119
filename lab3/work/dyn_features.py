"""
Script to build dynamic features + normalization
"""
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def stack_features(feature_mat, stack=3):
    """
    Args:
        feature_mat (ndarray): The feature matrix with shape = (ntime, n_features)
        stack: the number of stacks consecutive feature vectors.
            Notes that the stacking is assumed symmetric and therefore the output
            feature will be nfeatures*2*stack+1

    Return: stacked feature matrix with shape (ntime, nfeatures*(2*stack+1))
    """
    padded_features = np.pad(feature_mat, ((stack, stack), (0, 0)), 'reflect')

    stack_list = list()
    ntime = feature_mat.shape[0]
    for i in range(0, 2*stack+1):
        start_idx = i
        end_idx = start_idx + ntime
        stack_list.append(padded_features[start_idx:end_idx])

    stacked_mat = np.hstack(stack_list)
    return stacked_mat

if __name__ == "__main__":
    feature_name = 'lmfcc'
    # feature_name = 'mspec'

    print("[Dynamic feature] Working on the feature: ", feature_name)
    # load data
    data = np.load('data/train_val_data.npz')
    valdata = data['validation']
    traindata = data['train']
    testdata = np.load('data/testdata.npz')['testdata']

    # Calculate the mean and std first
    print("Calculate the mean and std ....")
    # for each utterance generate dynamic features
    scaler = StandardScaler()

    for d in tqdm(traindata):
        feature_mat = d[feature_name]
        stacked = stack_features(feature_mat)
        scaler.partial_fit(stacked) # do online fitting
    print("Complete calculate the standard scaler")
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
            dyn_feature = stack_features(d[feature_name])
            # use half precision
            new_data[feature_name] = scaler.transform(dyn_feature).astype("float16")
            data.append(new_data)

        print("[Dynamic feature] Complete normlaizating ", k)
        # save it
        np.savez('data/dyn/{}_{}.npz'.format(feature_name, k), data=data)
        print("[Dynamic feature] Wrote the normlaized data", k)

