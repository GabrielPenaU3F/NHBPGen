from abc import abstractmethod, ABC

import numpy as np

from domain.processes.nhbp import NHBP
from exceptions import ModelParametersException


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
        p = np.exp(-gamma * self.Kappa_s_t(0, t))
        return r * (1 - p) / p

    @abstractmethod
    def kappa_t(self, t):
        pass

    @abstractmethod
    def Kappa_s_t(self, s, t):
        pass

    def F_t(self, k, s, t):
        gamma, beta = self.model_params[:2]
        integral = self.Kappa_s_t(s, t)
        return 1 - np.exp(-(beta + gamma * k) * integral)

    def validate_model_parameters(self, model_params):
        gamma, beta = model_params
        if not gamma > 0:
            raise ModelParametersException('GPP gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('GPP beta parameter must be a positive number')


    def generate_next_arrival_time(self, current_state, present_time):
        return self.interarrival_inverse_cdf(current_state, present_time)

    @abstractmethod
    def interarrival_inverse_cdf(self, current_state, present_time):
        pass
