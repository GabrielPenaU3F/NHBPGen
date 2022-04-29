import unittest

from domain.polya_process import PolyaProcess
from exceptions import ModelParametersException


class PolyaTest(unittest.TestCase):

    def test_polya_process_receives_two_parameters(self):
        try:
            PolyaProcess((1, 2))
        except ModelParametersException:
            self.fail()

    def test_polya_process_cannot_receive_one_parameter(self):
        err_msg = 'Incorrect number of parameters for this model'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(1))

    def test_polya_process_alpha_parameter_cannot_be_negative_nor_zero(self):
        beta = 2
        err_msg = 'Polya alpha parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess((-1, beta)))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess((0, beta)))

    def test_polya_process_beta_parameter_cannot_be_negative_nor_zero(self):
        alpha = 2
        err_msg = 'Polya beta parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess((alpha, -1)))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess((alpha, 0)))


if __name__ == '__main__':
    unittest.main()
