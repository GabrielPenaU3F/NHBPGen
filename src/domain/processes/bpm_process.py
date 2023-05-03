from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class BPMProcess(GPP):

    def set_number_of_parameters(self):
        self.n_of_params = 2

    def kappa_t(self, t):
        gamma, beta = self.model_params
        return 1/(1 + beta * t)

    def validate_model_parameters(self, model_params):
        gamma, beta = model_params
        if not gamma > 0:
            raise ModelParametersException('BPM gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('BPM beta parameter must be a positive number')
        return gamma, beta
