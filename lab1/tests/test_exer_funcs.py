import unittest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from lab1_proto import enframe
from lab1_proto import preemp
from lab1_proto import windowing
from lab1_proto import powerSpectrum
from lab1_proto import logMelSpectrum
from lab1_proto import cepstrum
from lab1_tools import lifter
from lab1_proto import dtw
from lab1_proto import mfcc
import dtw as dtw_mod

class TestExerciseFunctions(unittest.TestCase):
    # setting
    win_size_s = 20e-3      # 20ms
    win_shift_s = 10e-3     # 10ms
    nceps = 13

    def setUp(self):
        self.example = np.load('data/lab1_example.npz')['example'].item()
        self.tidigits = np.load('data/lab1_data.npz')['data']

    def test_enframe(self):
        sampling_rate = self.example['samplingrate']
        winlen = int(self.win_size_s * sampling_rate)
        winshift = int(self.win_shift_s * sampling_rate)
        frames = enframe(self.example['samples'], winlen=winlen, winshift=winshift)
        exp_ans = self.example['frames']
        assert_allclose(frames, exp_ans)

    def test_preemp(self):
        pre_emph = preemp(self.example['frames'], p=0.97)
        exp_ans = self.example['preemph']
        assert_allclose(pre_emph, exp_ans)

    def test_windowing(self):
        windowed = windowing(self.example['preemph'])
        exp_ans = self.example['windowed']
        assert_allclose(windowed, self.example['windowed'])

    def test_powerSpectrum(self):
        exp_ans = self.example['spec']
        nfft = exp_ans.shape[1]
        power_spec = powerSpectrum(self.example['windowed'], nfft=nfft)
        assert_allclose(power_spec, exp_ans)

    def test_logMelSpectrum(self):
        mspec = logMelSpectrum(self.example['spec'], self.example['samplingrate'])
        exp_ans = self.example['mspec']
        assert_allclose(mspec, exp_ans)

    def test_cepstrum(self):
        mfcc = cepstrum(self.example['mspec'], nceps=self.nceps)
        assert_allclose(mfcc, self.example['mfcc'])

    def test_lmfcc(self):
        lmfcc = lifter(self.example['mfcc'])
        assert_allclose(lmfcc, self.example['lmfcc'])

    def test_dtw_ideal(self):
        x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
        dist = lambda x, y: np.linalg.norm(x-y, ord=2)

        d1, cost1, acc_cost1, path1 = dtw_mod.dtw(x, y, dist=dist)
        d2, cost2, acc_cost2, path2 = dtw(x, y, dist=dist, debug=True)

        assert_almost_equal(d1, d2)
        assert_allclose(cost1, cost2)
        assert_allclose(acc_cost1, acc_cost2)

        path_r_idx, path_c_idx = path1
        self.assertEqual(len(path2), len(path_r_idx))
        self.assertEqual(len(path2), len(path_c_idx))

        for idx, (i, j) in enumerate(path2):
            self.assertEqual(i, path_r_idx[idx])
            self.assertEqual(j, path_c_idx[idx])

    def test_dtw(self):
        x = mfcc(self.tidigits[12]['samples'])
        y = mfcc(self.tidigits[22]['samples'])
        dist = lambda x, y: np.linalg.norm(x-y, ord=2)
        d1, cost1, acc_cost1, path1 = dtw_mod.dtw(x, y, dist=dist)
        d2 = dtw(x, y, dist=dist, debug=False)
        assert_almost_equal(d1, d2)

    def test_dtw_zero(self):
        x = mfcc(self.tidigits[7]['samples'])
        dist = lambda x, y: np.linalg.norm(x-y, ord=2)
        d1, cost1, acc_cost1, path1 = dtw_mod.dtw(x, x, dist=dist)
        d2 = dtw(x, x, dist=dist, debug=False)
        assert_almost_equal(d1, d2)
        # assert_allclose(d2, 0.0)
        self.assertEqual(d2, 0.0)

    def test_dtw_symmetric(self):
        x = mfcc(self.tidigits[17]['samples'])
        y = mfcc(self.tidigits[3]['samples'])
        dist = lambda x, y: np.linalg.norm(x-y, ord=2)
        d1 = dtw(x, y, dist=dist, debug=False)
        d2 = dtw(y, x, dist=dist, debug=False)
        self.assertEqual(d2, d1)

