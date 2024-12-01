import numpy as np

from domain.processes.bpm_3p_process import BPM3pProcess
from exceptions import ModelParametersException


class BPMProcess(BPM3pProcess):

    def __init__(self, gamma, beta, initial_state=0):
        super().__init__(gamma, beta, beta, initial_state=initial_state)

    def kappa_t(self, t):
        gamma, beta = self.model_params[:2]
        return 1/(1 + beta * t)

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, beta = args
        return gamma, beta, beta

    def validate_model_parameters(self, model_params):
        gamma, beta, beta = model_params
        super().validate_model_parameters((gamma, beta, beta))
