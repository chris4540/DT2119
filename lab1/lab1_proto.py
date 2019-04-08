"""
DT2119, Lab 1 Feature Extraction

See also:
https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
import numpy as np
import scipy
import scipy.signal
from scipy import fftpack
from lab1_tools import trfbank
from lab1_tools import lifter

# Function given by the exercise ----------------------------------

def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    # consider the end of one frame as a pointer
    # 1. Subtract the first frame from the total length
    # 2. count number the number of frames obtained by shifting
    # 3. add back the step 1 frame
    num_frame = ((samples.shape[0] - winlen) // winshift) + 1

    ret = np.ndarray(shape=(num_frame, winlen))

    js = 0
    jn = js + winlen
    for i in range(num_frame):
        ret[i, :] = samples[js:jn]
        js += winshift
        jn = js + winlen

    return ret

def preemp(input_, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # y[n] = x[n] - p * x[n-1]
    b_coeff = np.array([1.0, -p])
    a_coeff = np.array([1.0])
    return scipy.signal.lfilter(b_coeff, a_coeff, input_, axis=1)


def windowing(input_):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)

    PS: We use hamming windows to reduce the spectral leakage as we do finite data fourier transform.
    See also:
    1. https://www.edn.com/electronics-news/4383713/Windowing-Functions-Improve-FFT-Results-Part-I
    2. https://en.wikipedia.org/wiki/Spectral_leakage
    """
    frame_size = input_.shape[1]  # We apply the hamming window to each frame
    return input_ * scipy.signal.hamming(frame_size, sym=False)

def powerSpectrum(input_, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    ret = np.abs(fftpack.fft(input_, nfft))**2
    return ret

def logMelSpectrum(input_, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank

    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input_.shape[1]
    return np.log(input_.dot(trfbank(samplingrate, nfft).T))

def cepstrum(input_, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # Lecture notes match only type II cosine transform
    ret = fftpack.dct(input_, type=2, axis=1)[:, :nceps]
    return ret

def dtw(x, y, dist=None):
    """
    Dynamic Time Warping

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.

    Impl. details:
    https://en.wikipedia.org/wiki/Dynamic_time_warping
    https://github.com/pierre-rouanet/dtw
    """
    # check args
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y should have the same 2nd dimension!")

    if dist is None:
        # default use Euclidean distances
        dist = lambda x, y: np.linalg.norm(x-y, ord=2)

    # obtain the dimensions
    N = x.shape[0]
    M = y.shape[0]
    D = x.shape[1]

    # calculate the local distacne matrix first
    loc_dist = np.full((N+1, M+1), inf)
    pass
