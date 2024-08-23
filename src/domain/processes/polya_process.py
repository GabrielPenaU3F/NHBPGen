import numpy as np

from domain.processes.bpm_3p_process import BPM3pProcess
from exceptions import ModelParametersException


class PolyaProcess(BPM3pProcess):

    def __init__(self, gamma, beta, initial_state=0):
        super().__init__(gamma, beta, gamma, initial_state=initial_state)

    def kappa_t(self, t):
        gamma, beta = self.model_params
        return 1/(1 + gamma * t)

    def Kappa_t(self, t):
        gamma, beta = self.model_params
        return (1/gamma) * np.log(1 + gamma * t)

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, gamma = args
        return gamma, beta

    def validate_model_parameters(self, model_params):
        gamma, beta = model_params
        if not gamma > 0:
            raise ModelParametersException('Polya gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('Polya beta parameter must be a positive number')
        return gamma, beta
