import numpy as np

from domain.processes.nhbp import NHBP
from exceptions import ModelParametersException


class PoissonProcess(NHBP):

    def generate_interarrival_time(self, current_state, present_time):
        lambda_ = self.model_params
        scale = 1/lambda_
        return np.random.exponential(scale)

    # Fix the number of parameters the particular model has
    def set_number_of_parameters(self):
        self.n_of_params = 1

    def intensity_function(self, r, t):
        return self.model_params

    def validate_model_parameters(self, model_params):
        lambda_ = model_params
        if not lambda_ > 0:
            raise ModelParametersException('Poisson lambda parameter must be a positive number')
        return lambda_
