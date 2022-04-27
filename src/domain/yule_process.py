from domain.nhbp import NHBP


class YuleProcess(NHBP):

    def __init__(self, model_params, initial_state):
        super().__init__(model_params, initial_state)

    def intensity_function(self, r, t):
        return self.model_params * r

    def set_number_of_parameters(self):
        self.n_of_params = 1
