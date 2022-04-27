from domain.nhbp import NHBP


class PoissonProcess(NHBP):

    def __init__(self, model_params, initial_state=0):
        super().__init__(model_params, initial_state=initial_state)

    # Fix the number of parameters the particular model has
    def set_number_of_parameters(self):
        self.n_of_params = 1

    def intensity_function(self, r, t):
        return self.model_params
