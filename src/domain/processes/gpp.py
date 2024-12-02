from abc import abstractmethod, ABC

import numpy as np
from scipy.integrate import quad

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

    # Overwrite this in subclasses if closed formulae are available
    def Kappa_s_t(self, s, t):
        integrando = lambda x: self.kappa_t(x)
        if np.isscalar(t):  # if t is a scalar
            return quad(integrando, s, t)[0]
        else:  # if t is an array
            result = np.zeros_like(t, dtype=float)
            for i, ti in enumerate(t):
                result[i] = quad(integrando, s, ti)[0]
            return result

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

    @abstractmethod
    def generate_next_arrival_time(self, current_state, present_time):
        pass
