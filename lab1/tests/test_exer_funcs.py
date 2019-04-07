import unittest
import numpy as np
from numpy.testing import assert_allclose
from lab1_proto import enframe
from lab1_proto import preemp


class TestExerciseFunctions(unittest.TestCase):
    # setting
    win_size_s = 20e-3      # 20ms
    win_shift_s = 10e-3     # 10ms

    def setUp(self):
        self.data = np.load('data/lab1_example.npz')['example'].item()
        self.samples = self.data['samples']

        sampling_rate = self.data['samplingrate']

        self.winlen = int(self.win_size_s * sampling_rate)
        self.winshift = int(self.win_shift_s * sampling_rate)

    def test_enframe(self):
        frames = enframe(self.samples, winlen=self.winlen, winshift=self.winshift)
        exp_ans = self.data['frames']
        assert_allclose(frames, exp_ans, rtol=0, atol=0)

    def test_preemp(self):
        frames = enframe(self.samples, winlen=self.winlen, winshift=self.winshift)
        pre_emph = preemp(frames, p=0.97)
        exp_ans = self.data['preemph']
        assert_allclose(pre_emph, exp_ans, rtol=0, atol=0)
