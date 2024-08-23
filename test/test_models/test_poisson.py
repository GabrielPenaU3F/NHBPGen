import unittest

from domain.processes.poisson_process import PoissonProcess
from exceptions import ModelParametersException


class PoissonTest(unittest.TestCase):

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
