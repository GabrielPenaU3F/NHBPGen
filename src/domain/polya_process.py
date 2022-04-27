from domain.nhbp import NHBP


class PolyaProcess(NHBP):

    def set_number_of_parameters(self):
        self.n_of_params = 1

    def intensity_function(self, r, t):
        alpha = self.model_params
        return (1 + alpha * r)/(1 + alpha * t)
