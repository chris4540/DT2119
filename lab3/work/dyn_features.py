"""
Script to build dynamic features + normalization
"""
import numpy as np
from tqdm import tqdm

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
    # feature_name = 'lmfcc'
    feature_name = 'mspec'

    print("[Dynamic feature] Working on the feature: ", feature_name)
    # load data
    data = np.load('data/train_val_data.npz')
    valdata = data['validation']
    traindata = data['train']
    testdata = np.load('data/testdata.npz')['testdata']

    # Calculate the mean and std first
    print("Calculate the mean and std ....")
    # for each utterance generate dynamic features
    train_features = []
    for d in tqdm(traindata):
        feature_mat = d[feature_name]
        stacked = stack_features(feature_mat)
        train_features.append(stacked)

    train_concat = np.concatenate(train_features, axis=0)
    # cal mean and std
    mean = np.mean(train_concat, axis=0)
    std = np.std(train_concat,  axis=0)
    print("Complete calculate the mean and std")
    train_concat = None
    train_features = None
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
            new_data['filename'] = d['filename']
            new_data['targets'] = d['targets']
            dyn_feature = stack_features(d[feature_name])
            new_data[feature_name] = (dyn_feature - mean) / std
            data.append(new_data)

        print("[Dynamic feature]Complete normlaizating ", k)
        # save it
        np.savez('data/dyn/{}_{}.npz'.format(feature_name, k), data=data)
        print("[Dynamic feature] Wrote the normlaized data", k)








