"""
Visualization code for part 5

Q: Are features correlated?
A: They are not correlated.

Q: Is the assumption of diagonal covariance matrices for Gaussian modelling justified?
A: From the graph we can see that the off-diagonal term are small and close to zero.
   Therefore, the gaussian modelling is justified.
"""
import numpy as np
from lab1_proto import mfcc
from lab1_proto import mspec
import matplotlib.pyplot as plt


def concat_all_features(data, feature="mfcc"):
    assert feature in ["mfcc", "mspec"]
    all_features = None
    for d in data:
        sample = d['samples']
        sampling_rate = d['samplingrate']
        if feature == "mfcc":
            features = mfcc(sample, samplingrate=sampling_rate)
        elif feature == "mspec":
            features = mspec(sample, samplingrate=sampling_rate)

        if all_features is None:
            all_features = features
        else:
            all_features = np.concatenate((all_features, features), axis=0)
    return all_features

def get_corrcoef_matrix(data, feature="mfcc"):
    all_features = concat_all_features(data, feature)
    return np.corrcoef(all_features.T)

if __name__ == "__main__":
    data = np.load('data/lab1_data.npz')['data']

    for f in ["mfcc", "mspec"]:
        corr = get_corrcoef_matrix(data, feature=f)
        # plot the matrix out
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ax.xaxis.tick_bottom()
        plt.title("Correlation coefficient matrix of %s" % f)
        plt.tight_layout()
        plt.savefig("part5_%s_cof_mat.png" % f)


