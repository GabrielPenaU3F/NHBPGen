from domain.nhbp import NHBP


class PolyaProcess(NHBP):

    def set_number_of_parameters(self):
        self.n_of_params = 2

    def intensity_function(self, r, t):
        alpha, beta = self.model_params
        return (alpha + beta * r)/(1 + beta * t)

    def validate_model_parameters(self, model_params):
        alpha, beta = model_params
        return alpha, beta