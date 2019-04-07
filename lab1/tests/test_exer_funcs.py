import unittest
import numpy as np
from numpy.testing import assert_allclose
from lab1_proto import enframe
from lab1_proto import preemp
from lab1_proto import windowing
from lab1_proto import powerSpectrum
from lab1_proto import logMelSpectrum
from lab1_proto import cepstrum
from lab1_tools import lifter

class TestExerciseFunctions(unittest.TestCase):
    # setting
    win_size_s = 20e-3      # 20ms
    win_shift_s = 10e-3     # 10ms
    nceps = 13

    def setUp(self):
        self.data = np.load('data/lab1_example.npz')['example'].item()
        self.samples = self.data['samples']

    def test_enframe(self):
        sampling_rate = self.data['samplingrate']
        winlen = int(self.win_size_s * sampling_rate)
        winshift = int(self.win_shift_s * sampling_rate)
        frames = enframe(self.samples, winlen=winlen, winshift=winshift)
        exp_ans = self.data['frames']
        assert_allclose(frames, exp_ans, rtol=0, atol=0)

    def test_preemp(self):
        # frames = enframe(self.samples, winlen=self.winlen, winshift=self.winshift)
        pre_emph = preemp(self.data['frames'], p=0.97)
        exp_ans = self.data['preemph']
        assert_allclose(pre_emph, exp_ans, rtol=0, atol=0)

    def test_windowing(self):
        windowed = windowing(self.data['preemph'])
        exp_ans = self.data['windowed']
        assert_allclose(windowed, self.data['windowed'], rtol=0, atol=0)

    def test_powerSpectrum(self):
        exp_ans = self.data['spec']
        nfft = exp_ans.shape[1]
        power_spec = powerSpectrum(self.data['windowed'], nfft=nfft)
        assert_allclose(power_spec, exp_ans, rtol=0, atol=0)

    def test_logMelSpectrum(self):
        mspec = logMelSpectrum(self.data['spec'], self.data['samplingrate'])
        exp_ans = self.data['mspec']
        assert_allclose(mspec, exp_ans, rtol=0, atol=0)

    def test_cepstrum(self):
        mfcc = cepstrum(self.data['mspec'], nceps=self.nceps)
        assert_allclose(mfcc, self.data['mfcc'], rtol=0, atol=0)

    def test_lmfcc(self):
        lmfcc = lifter(self.data['mfcc'])
        assert_allclose(lmfcc, self.data['lmfcc'], rtol=0, atol=0)