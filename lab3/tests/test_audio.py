"""
3 Data, test functions
"""
from lab1_proto import mfcc
from lab3_tools import loadAudio

if __name__ == "__main__":
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav'
    samples, samplingrate = loadAudio(filename)
    assert samples[0] == 11
    assert samples[1] == 13
    assert samplingrate == 20000
    # lmfcc = mfcc(samples)