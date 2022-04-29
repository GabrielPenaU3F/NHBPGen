import unittest

from domain.poisson_process import PoissonProcess
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


if __name__ == '__main__':
    unittest.main()
