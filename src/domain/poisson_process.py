from domain.nhbp import NHBP


class PoissonProcess(NHBP):

    # Fix the number of parameters the particular model has
    def set_number_of_parameters(self):
        self.n_of_params = 1

    def intensity_function(self, r, t):
        return self.model_params
