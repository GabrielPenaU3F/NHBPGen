import unittest

from domain.bpm_process import BPMProcess
from domain.polya_process import PolyaProcess
from exceptions import ModelParametersException


class BPMTest(unittest.TestCase):

    def test_bpm_process_receives_two_parameters(self):
        try:
            BPMProcess((1, 2))
        except ModelParametersException:
            self.fail()

    def test_bpm_process_cannot_receive_one_parameter(self):
        err_msg = 'Incorrect number of parameters for this model'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess(1))

    def test_bpm_process_alpha_parameter_cannot_be_negative_nor_zero(self):
        beta = 2
        err_msg = 'BPM alpha parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess((-1, beta)))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess((0, beta)))

    def test_bpm_process_beta_parameter_cannot_be_negative_nor_zero(self):
        alpha = 2
        err_msg = 'BPM beta parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess((alpha, -1)))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: BPMProcess((alpha, 0)))


if __name__ == '__main__':
    unittest.main()
