"""
Part 6. Explore Speech Segments with Clustering

This script plots the posterior probability matrices of the speechs of digit 1 and 7
with different hyperparamaters of GMM (i.e. the number of target classes)

The plot titles gives out the details

e.g. "gmm_post_prob_nc16_digit_7.png"
    plots the posterior probability matrices of the the speechs of digit 7
    given by the 16-class GMM
"""
from sklearn.mixture import GaussianMixture
from feature_corr import concat_all_features
from lab1_proto import mfcc
import numpy as np
import matplotlib.pyplot as plt
import os
import config

def pick_data_by_digit(data, digit):
    """
    Args:
        digit: the string of the target digit
    """
    ret = list()
    for d in data:
        if d['digit'] == digit:
            ret.append(d)
    return ret

if __name__ == "__main__":
    data = np.load(config.lab1_npz_file)['data']
    all_features = concat_all_features(data, feature="mfcc")

    try:
        os.makedirs("plots")
    except FileExistsError:
        pass

    for ncom in [4, 8, 16, 32]:
        clf = GaussianMixture(
            n_components=ncom, covariance_type='diag',
            random_state=40, verbose=1)  # fix the initialization for repetition

        # train the GMM with all data
        clf.fit(all_features)

        for digit in ['1', '4', '7']:
            test_data = pick_data_by_digit(data, digit=digit)

            fig, axes = plt.subplots(
                nrows=len(test_data), ncols=1, sharex=True, sharey=True,
                figsize=(12, 8))
            # prediction and plot the posterior matrix
            for i, d in enumerate(test_data):
                features = mfcc(d['samples'], samplingrate=d['samplingrate'])
                posterior_prob = clf.predict_proba(features)
                ax = axes[i]
                title = 'Posterior for digit {digit} by {speaker} ({gender})'.format(**d)
                ax.set_title(title)
                im = ax.matshow(posterior_prob.T)
                ax.xaxis.tick_bottom()
                ax.set_aspect('auto')

            for ax in axes.flat:
                ax.set(xlabel='frame', ylabel='classes')
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axes.flat:
                ax.label_outer()

            fig.tight_layout()
            # add shared colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            plt.savefig("./plots/gmm_post_prob_nc%d_digit_%s.png" % (ncom, digit),
                        dpi=100, bbox_inches='tight')
