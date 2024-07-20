import numpy as np

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class BPM3pProcess(GPP):

    def __init__(self, gamma, beta, rho, extra_params=tuple(), initial_state=0):
        extra_params = (rho,) + extra_params
        super().__init__(gamma, beta, extra_params, initial_state)

    def set_number_of_parameters(self):
        self.n_of_params = 3

    def kappa_t(self, t):
        gamma, beta, rho = self.model_params
        return 1/(1 + rho * t)

    def Kappa_t(self, t):
        gamma, beta, rho = self.model_params
        return (1/rho) * np.log(1 + rho * t)

    def validate_model_parameters(self, model_params):
        gamma, beta, rho = model_params
        if not gamma > 0:
            raise ModelParametersException('BPM-3p gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('BPM-3p beta parameter must be a positive number')
        if not rho > 0:
            raise ModelParametersException('BPM-3p rho parameter must be a positive number')
        return gamma, beta, rho

    def interarrival_inverse_cdf(self, x, k, s):
        gamma, beta, rho = self.model_params
        exponent = -rho / (beta + gamma * k)
        second_factor = np.power(1 - x, exponent)
        return ((1 + rho * s) * second_factor - 1) / beta
