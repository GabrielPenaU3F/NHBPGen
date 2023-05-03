import unittest

from domain.processes.poisson_process import PoissonProcess
from exceptions import ModelParametersException


class PoissonTest(unittest.TestCase):

    def test_poisson_process_receives_one_parameter(self):
        lambda_param = 1
        try:
            PoissonProcess(lambda_param)
        except ModelParametersException:
            self.fail()

    def test_poisson_process_cannot_receive_two_parameters(self):
        params = (1, 2)
        err_msg = 'Incorrect number of parameters for this model'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PoissonProcess(params))

    def test_poisson_process_parameter_cannot_be_negative(self):
        err_msg = 'Poisson lambda parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PoissonProcess(-5))

    def test_poisson_process_parameter_cannot_be_zero(self):
        err_msg = 'Poisson lambda parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PoissonProcess(0))


if __name__ == '__main__':
    unittest.main()
