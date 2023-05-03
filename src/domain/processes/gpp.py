from abc import abstractmethod

from domain.processes.nhbp import NHBP


class GPP(NHBP):

    def __init__(self, slope, intercept, extra_params=tuple(), initial_state=0):
        full_set_of_params = tuple((slope, intercept)) + extra_params
        self.slope = slope
        self.intercept = intercept
        super().__init__(full_set_of_params, initial_state)

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    def intensity_function(self, r, t):
        gamma, beta = self.slope, self.intercept
        return (beta + gamma * r) * self.kappa_t(t)

    @abstractmethod
    def kappa_t(self, t):
        pass

    def validate_model_parameters(self, model_params):
        pass

    