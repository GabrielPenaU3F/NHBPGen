from abc import ABC, abstractmethod

from exceptions import ModelParametersException


class NHBP(ABC):

    n_of_params = None
    model_params = None
    initial_state = None

    def __init__(self, model_params, initial_state):
        self.initial_state = initial_state
        self.set_number_of_parameters()
        self.check_number_of_parameters(model_params)
        self.model_params = model_params

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    @abstractmethod
    def intensity_function(self, r, t):
        pass

    def get_initial_state(self):
        return self.initial_state

    def check_number_of_parameters(self, model_params):
        if isinstance(model_params, int):
            if 1 != self.n_of_params:
                raise ModelParametersException('Incorrect number of parameters for this model')
        elif len(model_params) != self.n_of_params:
            raise ModelParametersException('Incorrect number of parameters for this model')

    def generate_sample_path(self, time, fs):
        pass
