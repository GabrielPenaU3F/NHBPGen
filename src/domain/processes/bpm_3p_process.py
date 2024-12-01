import numpy as np

from domain.processes.gpp import GPP
from exceptions import ModelParametersException
import numba

class BPM3pProcess(GPP):

    def __init__(self, gamma, beta, rho, initial_state=0):
        super().__init__(gamma, beta, rho, initial_state=initial_state)

    def kappa_t(self, t):
        gamma, beta, rho = self.model_params
        return 1/(1 + rho * t)

    def Kappa_s_t(self, s, t):
        gamma, beta, rho = self.model_params
        return (1/rho) * np.log((1 + rho * t)/(1 + rho * s))

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, rho = args
        return gamma, beta, rho

    def validate_model_parameters(self, model_params):
        gamma, beta, rho = model_params
        super().validate_model_parameters((gamma, beta))
        if not rho > 0:
            raise ModelParametersException('BPM-3p rho parameter must be a positive number')
        return gamma, beta, rho

    def interarrival_inverse_cdf(self, k, s):
        gamma, beta, rho = self.model_params
        return BPM3pProcess.numba_inverse_cdf(gamma, beta, rho, k, s)

    @staticmethod
    @numba.njit
    def numba_inverse_cdf(gamma, beta, rho, k, s):
        random = np.random.rand()
        exponent = -rho / (beta + gamma * k)
        second_factor = np.power(1 - random, exponent)
        return ((1 + rho * s) * second_factor - 1) / rho
