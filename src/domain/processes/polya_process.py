import numpy as np

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class PolyaProcess(GPP):

    def set_number_of_parameters(self):
        self.n_of_params = 2

    def kappa_t(self, t):
        gamma, beta = self.model_params
        return 1/(1 + gamma * t)

    def validate_model_parameters(self, model_params):
        gamma, beta = model_params
        if not gamma > 0:
            raise ModelParametersException('Polya gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('Polya beta parameter must be a positive number')
        return gamma, beta

    def generate_next_arrival_time(self, current_state, present_time):
        # gamma, beta = self.model_params
        # polya_interarrival = PolyaInterarrival(gamma, beta, current_state, present_time)
        # return polya_interarrival.rvs()
        random = np.random.rand()
        return self.interarrival_inverse_cdf(random, current_state, present_time)

    def interarrival_inverse_cdf(self, x, k, s):
        gamma, beta = self.model_params
        exponent = -gamma/(beta + gamma*k)
        second_factor = np.power(1 - x, exponent)
        return ((1 + gamma*s) * second_factor - 1)/gamma

