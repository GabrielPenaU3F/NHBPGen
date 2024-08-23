import unittest

from domain.processes.bpm_process import BPMProcess
from exceptions import ModelParametersException


class BPMTest(unittest.TestCase):

    def test_bpm_process_gamma_parameter_cannot_be_negative_nor_zero(self):
        beta = 2
        err_msg = 'BPM gamma parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess(-1, beta))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess(0, beta))

    def test_bpm_process_beta_parameter_cannot_be_negative_nor_zero(self):
        gamma = 2
        err_msg = 'BPM beta parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess(gamma, -1))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess(gamma, 0))


if __name__ == '__main__':
    unittest.main()
