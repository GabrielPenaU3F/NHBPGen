import numpy as np
import scipy

from domain.processes.fendick_process import FendickProcess
from domain.processes.polya_process import PolyaProcess
from domain.queues.arrival_adaptive.arrival_adaptive_queue import ArrivalAdaptiveQueue


class ArrivalAdaptivePolyaFendickNQueue(ArrivalAdaptiveQueue):

    def __init__(self, arrival_gamma, arrival_beta, service_gamma, service_beta, fendick_rho, n_servers=1):
        arrival_process = PolyaProcess(arrival_gamma, arrival_beta)
        service_process = FendickProcess(service_gamma, service_beta, fendick_rho)
        super().__init__(arrival_process, service_process, n_servers)

    def generate_next_transition_time(self, cumulative_arrivals, present_time):
        random = np.random.rand()
        return scipy.optimize.fsolve(self.implicit_inverse_cdf, x0=present_time,
                                     args=(random, cumulative_arrivals, present_time))

    def implicit_inverse_cdf(self, t, *args):
        #   Here, gamma_1 and beta_1 are gamma/rho and beta/rho, respectively
        random, k, s = args
        arrival_gamma, arrival_beta = self.arrival_process.get_model_parameters()
        service_gamma_1, service_beta_1, rho = self.service_process.get_model_parameters()
        first_exponent = -(arrival_beta + arrival_gamma*k)/arrival_gamma
        second_exponent = -(service_beta_1 + service_gamma_1*k)/service_gamma_1
        first_factor = ((1 + arrival_gamma*t)/(1 + arrival_gamma*s)) ** first_exponent
        service_gamma = service_gamma_1 * rho
        second_factor = ((1 + service_gamma*t)/(1 + service_gamma*s)) ** second_exponent
        return first_factor * second_factor - (1 - random)
