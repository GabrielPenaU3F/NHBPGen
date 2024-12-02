import numpy as np

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class FendickProcess(GPP):

    def __init__(self, gamma, beta, rho, initial_state=0):
        super().__init__(gamma/rho, beta/rho,rho, initial_state=initial_state)

    def kappa_t(self, t):
        gamma_per_rho, beta_per_rho, rho = self.model_params
        gamma = gamma_per_rho * rho
        return 1/(1 + gamma * t)

    def Kappa_t(self, t):
        gamma_per_rho, beta_per_rho, rho = self.model_params
        gamma = gamma_per_rho * rho
        return (1/gamma) * np.log(1 + gamma * t)

    def determine_mandatory_parameters(self, *args, **kwargs):
        return args

    def validate_model_parameters(self, model_params):
        gamma_per_rho, beta_per_rho, rho = model_params
        if not gamma_per_rho > 0:
            raise ModelParametersException('Fendick slope parameter must be a positive number')
        if not beta_per_rho > 0:
            raise ModelParametersException('Fendick intercept parameter must be a positive number')
        if not rho > 0:
            raise ModelParametersException('Fendick rho parameter must be a positive number')
        return gamma_per_rho, beta_per_rho, rho

    def generate_next_arrival_time(self, k, s):
        random = np.random.rand()
        gamma_per_rho, beta_per_rho, rho = self.model_params
        exponent = -gamma_per_rho/(beta_per_rho + gamma_per_rho*k)
        second_factor = np.power(1 - random, exponent)
        gamma = gamma_per_rho * rho
        return ((1 + gamma*s) * second_factor - 1)/gamma
