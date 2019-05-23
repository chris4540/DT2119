"""
Script to build dynamic features
"""
import numpy as np

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
    pass