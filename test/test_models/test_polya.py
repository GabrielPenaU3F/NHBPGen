import unittest

from domain.processes.polya_process import PolyaProcess
from exceptions import ModelParametersException


class PolyaTest(unittest.TestCase):

    def test_polya_process_gamma_parameter_cannot_be_negative_nor_zero(self):
        beta = 2
        err_msg = 'Polya gamma parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(-1, beta))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(0, beta))

    def test_polya_process_beta_parameter_cannot_be_negative_nor_zero(self):
        gamma = 2
        err_msg = 'Polya beta parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(gamma, -1))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(gamma, 0))


if __name__ == '__main__':
    unittest.main()
