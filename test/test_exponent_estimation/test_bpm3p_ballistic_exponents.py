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

    def test_moses_calculation_must_fail_if_tmin_is_smaller_than_the_timegap(self):
        min_T = 10
        max_T = 8000
        delta = 50
        time_step = 1
        data_polya = BPM3pBallisticExponentsTest.ballistic_data
        err_msg = 'The delta time gap must be smaller than min_T'
        self.assertRaisesRegex(IndexError, err_msg,
                               lambda: BPM3pBallisticExponentsTest.et_averager.estimate_moses(
                                   data_polya, min_T, max_T, delta, time_step))

    def test_moses_exponent_must_be_one_half_for_polya_process(self):
        min_T = 100
        max_T = 8000
        delta = 10
        time_step = 1
        data_polya = BPM3pBallisticExponentsTest.ballistic_data
        moses = BPM3pBallisticExponentsTest.et_averager.estimate_moses(data_polya, min_T, max_T, delta, time_step)
        testing.assert_almost_equal(moses, 1/2, decimal=2)

if __name__ == '__main__':
    unittest.main()
