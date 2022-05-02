from domain.nhbp import NHBP
from exceptions import ModelParametersException


class YuleProcess(NHBP):

    def __init__(self, model_params, initial_state):
        super().__init__(model_params, initial_state)

    def intensity_function(self, r, t):
        return self.model_params * r

    def set_number_of_parameters(self):
        self.n_of_params = 1

    def validate_model_parameters(self, model_params):
        lambda_ = model_params
        if not lambda_ > 0:
            raise ModelParametersException('Yule parameter must be a positive number')
