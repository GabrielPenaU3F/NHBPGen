import unittest

from domain.poisson_process import PoissonProcess
from exceptions import ModelParametersException


class PoissonTest(unittest.TestCase):

    process = None

    def test_poisson_process_receives_one_parameter(self):
        lambda_param = 1
        try:
            PoissonProcess(lambda_param)
        except ModelParametersException:
            self.fail()

    def test_poisson_process_cannot_receive_two_parameters(self):
        params = (1, 2)
        err_msg = 'Incorrect number of parameters for this model'
        self.assertRaisesRegex(ModelParametersException, err_msg, PoissonProcess, params)

    def test_initial_population_is_0_if_not_specified(self):
        process = PoissonProcess(1)
        self.assertEqual(0, process.get_initial_state())

    def test_initial_population_is_10_when_specified(self):
        process = PoissonProcess(1, initial_state=10)
        self.assertEqual(10, process.get_initial_state())


if __name__ == '__main__':
    unittest.main()
