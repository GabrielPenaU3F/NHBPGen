from abc import abstractmethod

import numpy as np

from domain.processes.nhbp import NHBP


class PureBirth(NHBP):

    def __init__(self, model_params, initial_state=0):
        super().__init__(model_params, initial_state)

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    @abstractmethod
    def intensity_function(self, r, t):
        pass

    def validate_model_parameters(self, model_params):
        pass

    def generate_interarrival_time(self, current_state, present_time):
        updated_lambda = self.intensity_function(current_state, -1)
        scale = 1/updated_lambda
        return np.random.exponential(scale)

