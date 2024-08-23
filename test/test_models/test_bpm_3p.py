import unittest

from domain.processes.bpm_3p_process import BPM3pProcess
from exceptions import ModelParametersException


class BPM3pTest(unittest.TestCase):

    def test_bpm_3p_process_gamma_parameter_cannot_be_negative_nor_zero(self):
        beta = 2
        rho = 1
        err_msg = 'BPM-3p gamma parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(-1, beta, rho))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(0, beta, rho))

    def test_bpm_3p_process_beta_parameter_cannot_be_negative_nor_zero(self):
        gamma = 2
        rho = 1
        err_msg = 'BPM-3p beta parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(gamma, -1, rho))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(gamma, 0, rho))

    def test_bpm_3p_process_rho_parameter_cannot_be_negative_nor_zero(self):
        gamma = 2
        beta = 1
        err_msg = 'BPM-3p rho parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(gamma, beta, -1))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPM3pProcess(gamma, beta, 0))


if __name__ == '__main__':
    unittest.main()
