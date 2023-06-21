import numpy as np
import scipy

from domain.processes.polya_process import PolyaProcess
from domain.queues.arrival_adaptive.arrival_adaptive_queue import ArrivalAdaptiveQueue


class ArrivalAdaptivePolyaPolyaNQueue(ArrivalAdaptiveQueue):

    def __init__(self, arrival_gamma, arrival_beta, service_gamma, service_beta, n_servers=1):
        arrival_process = PolyaProcess(arrival_gamma, arrival_beta)
        service_process = PolyaProcess(service_gamma, service_beta)
        super().__init__(arrival_process, service_process, n_servers)

    def generate_next_transition_time(self, cumulative_arrivals, present_time):
        random = np.random.rand()
        return scipy.optimize.fsolve(self.implicit_inverse_cdf, x0=present_time,
                                     args=(random, cumulative_arrivals, present_time))

    def implicit_inverse_cdf(self, t, *args):
        random, k, s = args
        arrival_gamma, arrival_beta = self.arrival_process.get_model_parameters()
        service_gamma, service_beta = self.service_process.get_model_parameters()
        first_exponent = -(arrival_beta + arrival_gamma*k)/arrival_gamma
        second_exponent = -(service_beta + service_gamma*k)/service_gamma
        first_factor = ((1 + arrival_gamma*t)/(1 + arrival_gamma*s)) ** first_exponent
        second_factor = ((1 + service_gamma*t)/(1 + service_gamma*s)) ** second_exponent
        return first_factor * second_factor - (1 - random)
