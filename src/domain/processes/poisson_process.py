import numpy as np

from domain.processes.nhbp import NHBP
from exceptions import ModelParametersException


class PoissonProcess(NHBP):

    def __init__(self, lambda_, initial_state=0):
        super().__init__(lambda_, initial_state=initial_state)

    def generate_next_arrival_time(self, current_state, present_time):
        lambda_ = self.model_params
        scale = 1/lambda_
        return present_time + np.random.exponential(scale)

    def intensity_function(self, r, t):
        return self.model_params

    def determine_mandatory_parameters(self, *args, **kwargs):
        lambda_ = args[0]
        return lambda_

    def validate_model_parameters(self, model_params):
        lambda_ = model_params
        if not lambda_ > 0:
            raise ModelParametersException('Poisson lambda parameter must be a positive number')
        return lambda_

    def mean_value_function(self, t, lambda_):
        return lambda_ * t
