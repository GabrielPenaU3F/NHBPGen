from domain.processes.pure_birth import PureBirth
from exceptions import ModelParametersException


class YuleProcess(PureBirth):

    def __init__(self, a, initial_state):
        super().__init__(a, initial_state=initial_state)

    def intensity_function(self, r, t):
        a = self.model_params
        return a * r

    def set_number_of_parameters(self):
        self.n_of_params = 1

    def validate_model_parameters(self, model_params):
        lambda_ = model_params
        if not lambda_ > 0:
            raise ModelParametersException('Yule parameter must be a positive number')
        return lambda_
