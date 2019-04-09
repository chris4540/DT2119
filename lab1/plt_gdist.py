import numpy as np
import matplotlib.pyplot as plt
from lab1_tools import tidigit2labels
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

if __name__ == '__main__':
    gdist =  np.load('./data/global_dist.npy')
    data = np.load('data/lab1_data.npz')['data']
    # plt.matshow(mat)

    labels = tidigit2labels(data)
    fig = plt.figure(figsize=(25, 10))
    Z = linkage(gdist, "complete")
    dn = dendrogram(Z, labels=labels)
    fig.savefig('plots/linkage.png',dpi=80, bbox_inches='tight')
