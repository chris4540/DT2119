"""
This script is for the lab1 section 4:
    Mel Frequency Cepstrum Coefficients step-by-step
"""
import os
import numpy as np
import scipy
from scipy.fftpack import fft, fftshift
import scipy.signal
import lab1_proto
import matplotlib.pyplot as plt
import config
from lab1_tools import trfbank

if __name__ == '__main__':
    example = np.load(config.sample_npz_file)['example'].item()
    fig_size = [10, 1.5]

    # mkdir -p plots
    try:
        os.makedirs("plots")
    except FileExistsError:
        pass
    # ===================================================================
    # 4.1 Enframe
    win_size_s = 20e-3      # 20ms
    win_shift_s = 10e-3     # 10ms
    sampling_rate = example['samplingrate']
    winlen = int(win_size_s * sampling_rate)
    winshift = int(win_shift_s * sampling_rate)
    enframed = lab1_proto.enframe(example['samples'], winlen=winlen, winshift=winshift)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title("Enframed samples")
    ax.pcolormesh(enframed.T)
    fig.savefig("./plots/enframe.png", bbox_inches='tight')
    # ===================================================================
    # 4.3
    plt.figure()
    windown_length = example['preemph'].shape[1]
    window = scipy.signal.hamming(windown_length, sym=False)
    plt.plot(window)
    plt.title("Hamming window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.savefig("./plots/hamming.png", bbox_inches='tight')

    # plot fft of the window
    plt.figure()
    A = fft(window) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = np.abs(fftshift(A / abs(A).max()))
    plt.plot(freq, response)
    plt.title("Frequency response of the Hamming window")
    plt.ylabel("Frequency space amplitude")
    plt.xlabel("Frequency space sample")
    plt.savefig("./plots/hamming_fft.png", bbox_inches='tight')
    # ===================================================================
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title("Power spectrum")
    ax.pcolormesh(example['spec'].T)
    fig.savefig("./plots/power_spectrum.png", bbox_inches='tight')

    # ===========================================================
    # 4.5
    filter_ = trfbank(example['samplingrate'], 512)
    plt.figure(figsize=(12, 2))
    plt.xlim(0, 180)
    for cap in filter_:
        plt.plot(cap)
    plt.title("Mel filterbank")
    plt.savefig("./plots/mel_filterbank.png", bbox_inches='tight')

    #
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title("log-scale Mel spectrum")
    ax.pcolormesh(example['mspec'].T)
    fig.savefig("./plots/log_mel_spectrum.png", bbox_inches='tight')
    # ====================================================================
    # 4.6
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title("MFCCs")
    ax.pcolormesh(example['mfcc'].T)
    fig.savefig("./plots/mfcc.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title("Liftered MFCCs")
    ax.pcolormesh(example['lmfcc'].T)
    fig.savefig("./plots/lmfcc.png", bbox_inches='tight')
