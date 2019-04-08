"""
Calculate a global distance matrix and save it
"""
import numpy as np
from lab1_proto import dtw
from lab1_proto import mfcc
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = np.load('data/lab1_data.npz')['data']

    # calcualte the global distance matrix
    ndata = len(data)

    # global_dist matrix is symmetric
    # the diagonal terms are zeros
    global_dist = np.zeros((ndata, ndata))

    for i, j in zip(*np.triu_indices(ndata, k=1)):
        feature_i = mfcc(data[i]['samples'])
        feature_j = mfcc(data[j]['samples'])
        d = dtw(feature_i, feature_j)
        print(d)
        global_dist[i, j] = d
        global_dist[j, i] = d

    np.save("data/global_dist.npy", global_dist)
