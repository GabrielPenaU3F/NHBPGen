from abc import abstractmethod, ABC

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gam

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

    def log_likelihood(self, k, t):
        k = np.asarray(k)
        r = np.asarray(self.r())
        p = self.p(t)

        log_binom_part = np.zeros_like(k)
        for i in range(len(k)):
            print(f'Sample number {i}')
            if k[i] > 0:
                j = np.arange(k[i])
                log_binom_part[i] = np.sum(np.log(r + j))
            else:
                log_binom_part[i] = 0.0  # this if k=0

        log_fact_k = self.log_factorial(k)
        log_probs = (log_binom_part - log_fact_k + r * np.log(1 - p) + k * np.log(p))
        return np.sum(log_probs)

    def mean_value(self, t):
        r = self.r()
        p = self.p(t)
        return self.initial_state + r * (1 - p) / p

    def variance(self, t):
        r = self.r()
        p = self.p(t)
        return r * (1 - p) / (p**2)

    def r(self):
        gamma, beta = self.slope, self.intercept
        return beta/gamma

    def p(self, t):
        gamma = self.slope
        return np.exp(-gamma * self.Kappa_s_t(0, t))

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

    def log_factorial(self, k):
        k = np.asarray(k)
        result = np.zeros_like(k, dtype=np.float64)

        # Mask for small values
        small_k = k < 20
        large_k = ~small_k

        # Exact
        if np.any(small_k):
            from scipy.special import gammaln
            result[small_k] = gammaln(k[small_k] + 1)

        # Stirling approximation
        if np.any(large_k):
            k_large = k[large_k].astype(np.float64)
            result[large_k] = (
                    k_large * np.log(k_large)
                    - k_large
                    + 0.5 * np.log(2 * np.pi * k_large)
            )

        return result
