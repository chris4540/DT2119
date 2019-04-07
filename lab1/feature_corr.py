"""
For section 5
"""
import numpy as np
from lab1_proto import mfcc
import matplotlib.pyplot as plt

def get_corrcoef_matrix(data):
    all_features = np.empty((0,13))
    for d in data:
        sample = d['samples']
        sampling_rate = d['samplingrate']
        features = mfcc(sample)
        all_features = np.concatenate((all_features, features), axis=0)

    return np.corrcoef(all_features.T)

if __name__ == "__main__":
    data = np.load('data/lab1_data.npz')['data']
    corr = get_corrcoef_matrix(data)

    # plot the matrix out
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    plt.title("Correlation coefficient matrix")
    plt.savefig("part5_cof_mat.png")
