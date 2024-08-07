import unittest

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from exceptions import ModelParametersException, SimulationException


class NHBPTest(unittest.TestCase):

    def test_poisson_initial_population_is_0_if_not_specified(self):
        process = PoissonProcess(1)
        self.assertEqual(0, process.get_initial_state())

    def test_poisson_initial_population_is_10_when_specified(self):
        process = PoissonProcess(1, initial_state=10)
        self.assertEqual(10, process.get_initial_state())

    def test_polya_initial_population_is_0_if_not_specified(self):
        process = PolyaProcess(1, 2)
        self.assertEqual(0, process.get_initial_state())

    def test_polya_initial_population_is_10_when_specified(self):
        process = PolyaProcess(1, 2, initial_state=10)
        self.assertEqual(10, process.get_initial_state())

    def test_poisson_initial_population_cannot_be_negative(self):
        err_msg = 'Initial state must be a positive integer'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PoissonProcess(1, initial_state=-10))

    def test_polya_initial_population_cannot_be_negative(self):
        err_msg = 'Initial state must be a positive integer'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(1, 2, initial_state=-10))

    def test_poisson_initial_population_cannot_be_non_integer(self):
        err_msg = 'Initial state must be a positive integer'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PoissonProcess(1, initial_state=3.5))

    def test_polya_initial_population_cannot_be_non_integer(self):
        err_msg = 'Initial state must be a positive integer'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: PolyaProcess(1, 2, initial_state=3.5))


if __name__ == '__main__':
    unittest.main()
