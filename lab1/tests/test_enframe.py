import unittest
import numpy as np
from numpy.testing import assert_allclose
from lab1_proto import enframe


class TestEnframe(unittest.TestCase):
    # setting
    win_size_s = 20e-3      # 20ms
    win_shift_s = 10e-3     # 10ms

    def setUp(self):
        example = np.load('data/lab1_example.npz')['example'].item()
        self.samples = example['samples']
        self.frames = example['frames']
        sampling_rate = example['samplingrate']

        self.winlen = int(self.win_size_s * sampling_rate)
        self.winshift = int(self.win_shift_s * sampling_rate)

    def test_enframe(self):
        frames = enframe(self.samples, winlen=self.winlen, winshift=self.winshift)
        assert_allclose(frames, self.frames, rtol=0, atol=0)
