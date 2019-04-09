"""
Calculate a global distance matrix and save the gdist file
"""
import numpy as np
import config
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

    cnt = 0
    for i, j in zip(*np.triu_indices(ndata, k=1)):
        feature_i = mfcc(data[i]['samples'])
        feature_j = mfcc(data[j]['samples'])
        d = dtw(feature_i, feature_j)
        global_dist[i, j] = d
        global_dist[j, i] = d
        cnt += 1
        if cnt % 100 == 0:
            print("Calculated %d global distances" % cnt, flush=True)


    np.save(config.gdist_npy_file, global_dist)
