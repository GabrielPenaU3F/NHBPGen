from abc import abstractmethod

import numpy as np

from domain.processes.nhbp import NHBP


class GPP(NHBP):

    def __init__(self, slope, intercept, extra_params=tuple(), initial_state=0):
        full_set_of_params = tuple((slope, intercept)) + extra_params
        self.slope = slope
        self.intercept = intercept
        super().__init__(full_set_of_params, initial_state)

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    def intensity_function(self, k, t):
        gamma, beta = self.slope, self.intercept
        return (beta + gamma * k) * self.kappa_t(t)

    def mean_value(self, t):
        gamma, beta = self.slope, self.intercept
        r = beta/gamma
        p = np.exp(-gamma * self.Kappa_t(t))
        return r * (1 - p) / p

    @abstractmethod
    def kappa_t(self, t):
        pass

    @abstractmethod
    def Kappa_t(self, t):
        pass

    def validate_model_parameters(self, model_params):
        pass

    def generate_next_arrival_time(self, current_state, present_time):
        random = np.random.rand()
        return self.interarrival_inverse_cdf(random, current_state, present_time)

    @abstractmethod
    def interarrival_inverse_cdf(self, random_number, current_state, present_time):
        pass
