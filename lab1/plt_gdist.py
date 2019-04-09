"""
Plot and compare for questions 7

Question 7: Comparing Utterances

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from lab1_tools import tidigit2labels
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import itertools
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

    # generate the labels
    labels = tidigit2labels(data)
    for i, l in enumerate(labels):
        print(i, l)

    # plot the matrix
    fig, ax = plt.subplots(1, 1)
    cax = ax.matshow(gdist)
    ax.set_title("The global distance matrix")
    fig.colorbar(cax)
    fig.tight_layout()
    plt.savefig('plots/gdist_mat.png', dpi=100, bbox_inches='tight')

    # compare digit 7 for different speakers
    for i, j in itertools.combinations([16, 17, 38, 39], 2):
        print(i, j, labels[i], labels[j], gdist[i, j])


    # plot the dendrogram
    fig = plt.figure(figsize=(25, 10))
    Z = linkage(gdist, "complete")
    dn = dendrogram(Z, labels=labels)
    fig.savefig('plots/linkage.png', dpi=200, bbox_inches='tight')
