from domain.nhbp import NHBP


class BPMProcess(NHBP):

    def set_number_of_parameters(self):
        self.n_of_params = 2

    def intensity_function(self, r, t):
        alpha, beta = self.model_params
        return (alpha + beta * r)/(1 + alpha * t)
