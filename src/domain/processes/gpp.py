from abc import abstractmethod, ABC

import numpy as np

from domain.processes.nhbp import NHBP


class GPP(NHBP, ABC):

    def __init__(self, slope, intercept, *args, **kwargs):
        self.slope = slope
        self.intercept = intercept
        super().__init__(slope, intercept, *args, **kwargs)

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

    def generate_next_arrival_time(self, current_state, present_time):
        return self.interarrival_inverse_cdf(current_state, present_time)

    @abstractmethod
    def interarrival_inverse_cdf(self, current_state, present_time):
        pass
