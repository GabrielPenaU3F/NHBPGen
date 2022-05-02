from domain.nhbp import NHBP
from exceptions import ModelParametersException


class BPMProcess(NHBP):

    def set_number_of_parameters(self):
        self.n_of_params = 2

    def intensity_function(self, r, t):
        alpha, beta = self.model_params
        return (alpha + beta * r)/(1 + alpha * t)

    def validate_model_parameters(self, model_params):
        alpha, beta = model_params
        if not alpha > 0:
            raise ModelParametersException('BPM alpha parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('BPM beta parameter must be a positive number')
        return alpha, beta
