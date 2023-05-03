import unittest

from domain.processes.yule_process import YuleProcess
from exceptions import ModelParametersException


class YuleTest(unittest.TestCase):

    def test_yule_process_receives_two_parameters_and_initial_state(self):
        try:
            YuleProcess(1, 2)
        except ModelParametersException:
            self.fail()

    def test_yule_process_parameter_cannot_be_negative_nor_zero(self):
        err_msg = 'Yule parameter must be a positive number'
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: YuleProcess(-2, 1))
        self.assertRaisesRegex(ModelParametersException, err_msg,
                               lambda: YuleProcess(0, 1))


if __name__ == '__main__':
    unittest.main()
