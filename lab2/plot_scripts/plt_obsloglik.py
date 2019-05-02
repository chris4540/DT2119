"""
Run this script by:
    ipython plot_scripts/plt_obsloglik.py
    %run plot_scripts/plt_obsloglik.py
"""
import numpy as np
from lab2_tools import log_multivariate_normal_density_diag
import matplotlib.pyplot as plt
from prondict import isolated
from lab2_proto import concatHMMs



if __name__ == '__main__':

    # load data
    data = np.load('data/lab2_data.npz')['data']
    example = np.load('data/lab2_example.npz')['example'].item()
    phoneHMMs = np.load('data/lab2_models_onespkr.npz')['phoneHMMs'].item()


    # Build hmm
    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])
    wordHMMs['7'] = concatHMMs(phoneHMMs, isolated['7'])

    # caculate the obsloglik
    obsloglik = log_multivariate_normal_density_diag(
        example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    # Plot for o
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.pcolormesh(obsloglik.T)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig("./plots/obsloglik_o_woman.png",
                dpi=80, bbox_inches='tight')

    # caculate the obsloglik
    obsloglik = log_multivariate_normal_density_diag(
        data[22]['lmfcc'], wordHMMs['7']['means'], wordHMMs['7']['covars'])
    # Plot for o
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.pcolormesh(obsloglik.T)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig("./plots/obsloglik_7_woman.png",
                dpi=80, bbox_inches='tight')

