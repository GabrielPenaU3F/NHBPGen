from domain.processes.pure_birth import PureBirthProcess
from exceptions import ModelParametersException


class YuleProcess(PureBirthProcess):


    def __init__(self, a, initial_state):
        super().__init__(a, initial_state=initial_state)

    def intensity_function(self, k, t):
        a = self.model_params
        return a * k

    def determine_mandatory_parameters(self, *args, **kwargs):
        a = args[0]
        return a

    def validate_model_parameters(self, model_params):
        lambda_ = model_params
        if not lambda_ > 0:
            raise ModelParametersException('Yule parameter must be a positive number')
        return lambda_
