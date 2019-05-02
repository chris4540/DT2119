"""

Run this test with:
    nosetests -v tests/test_lab2_proto.py
"""
import unittest
import numpy as np
from lab2_tools import log_multivariate_normal_density_diag
from prondict import isolated
from lab2_proto import concatHMMs
from numpy.testing import assert_allclose

class TestExerciseFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load data
        cls.example = np.load('lab2_example.npz')['example'].item()
        cls.phoneHMMs = np.load('lab2_models_onespkr.npz')['phoneHMMs'].item()

    def test_concatHMMs(self):
        """
        test_concatHMMs: Required verification in section 5.1
        """
        # obtain the concated HMM model
        wordHMMs = {}
        wordHMMs['o'] = concatHMMs(self.phoneHMMs, isolated['o'])
        means = wordHMMs['o']['means']
        covars = wordHMMs['o']['covars']

        # use example observation to calculate the log likelihood of observation
        obsloglik = log_multivariate_normal_density_diag(self.example['lmfcc'], means, covars)

        # check against with the example
        assert_allclose(obsloglik, self.example['obsloglik'])
