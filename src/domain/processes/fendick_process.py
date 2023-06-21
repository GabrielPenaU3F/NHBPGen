import numpy as np

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class FendickProcess(GPP):

    def __init__(self, gamma, beta, rho):
        rho_tpl = (rho,)
        super().__init__(gamma/rho, beta/rho, extra_params=rho_tpl)

    def set_number_of_parameters(self):
        self.n_of_params = 3

    def kappa_t(self, t):
        gamma_per_rho, beta_per_rho, rho = self.model_params
        gamma = gamma_per_rho * rho
        return 1/(1 + gamma * t)

    def validate_model_parameters(self, model_params):
        gamma_per_rho, beta_per_rho, rho = model_params
        if not gamma_per_rho > 0:
            raise ModelParametersException('Fendick slope parameter must be a positive number')
        if not beta_per_rho > 0:
            raise ModelParametersException('Fendick intercept parameter must be a positive number')
        if not rho > 0:
            raise ModelParametersException('Fendick rho parameter must be a positive number')
        return gamma_per_rho, beta_per_rho, rho

    def interarrival_inverse_cdf(self, x, k, s):
        gamma_per_rho, beta_per_rho, rho = self.model_params
        exponent = -gamma_per_rho/(beta_per_rho + gamma_per_rho*k)
        second_factor = np.power(1 - x, exponent)
        gamma = gamma_per_rho * rho
        return ((1 + gamma*s) * second_factor - 1)/gamma
