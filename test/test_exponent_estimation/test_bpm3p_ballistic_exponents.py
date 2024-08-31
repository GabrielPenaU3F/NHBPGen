import unittest

import numpy as np
from numpy import testing

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager


class BPM3pBallisticExponentsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ensemble_averager = EnsembleAverager()
        cls.et_averager = EnsembleTimeAverager()
        cls.ballistic_data = np.load('./test_data/test_ensemble_ballisticdif.npz')['ensemble']

    def test_hurst_exponent_must_be_1_for_ballistic_diffusion(self):
        hurst = BPM3pBallisticExponentsTest.ensemble_averager.estimate_hurst(BPM3pBallisticExponentsTest.ballistic_data, 8000, 1)
        testing.assert_almost_equal(hurst, 1, decimal=3)


    def test_moses_exponent_must_be_one_half_for_polya_process(self):
        max_T = 8000
        delta = 10
        time_step = 1
        data_polya = BPM3pBallisticExponentsTest.ballistic_data
        moses = BPM3pBallisticExponentsTest.et_averager.estimate_moses(data_polya, max_T, delta, time_step)
        testing.assert_almost_equal(moses, 1/2, decimal=2)

    def test_noah_exponent_must_be_one_half_for_polya_process(self):
        max_T = 8000
        delta = 10
        time_step = 1
        M = 1/2
        data_polya = BPM3pBallisticExponentsTest.ballistic_data
        noah = BPM3pBallisticExponentsTest.et_averager.estimate_noah(data_polya, M, max_T, delta, time_step)
        testing.assert_almost_equal(noah, 1/2, decimal=2)

if __name__ == '__main__':
    unittest.main()
