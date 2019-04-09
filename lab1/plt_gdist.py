import os
import numpy as np
import matplotlib.pyplot as plt
from lab1_tools import tidigit2labels
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import config

if __name__ == '__main__':
    # mkdir -p plots
    try:
        os.makedirs("plots")
    except FileExistsError:
        pass

    # load data files
    gdist =  np.load(config.gdist_npy_file)
    data = np.load(config.lab1_npz_file)['data']

    # plot the matrix
    fig, ax = plt.subplots(1, 1)
    cax = ax.matshow(gdist)
    fig.colorbar(cax)
    plt.savefig('plots/gdist_mat.png', dpi=200, bbox_inches='tight')


    # plot the dendrogram
    labels = tidigit2labels(data)
    fig = plt.figure(figsize=(25, 10))
    Z = linkage(gdist, "complete")
    dn = dendrogram(Z, labels=labels)
    fig.savefig('plots/linkage.png', dpi=200, bbox_inches='tight')
