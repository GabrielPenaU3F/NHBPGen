from abc import ABC, abstractmethod

from scipy.optimize import curve_fit

class FitterModel(ABC):

    def fit(self, t, y, p0, initial_state=0):
        self.initial_state = initial_state
        params, cov = curve_fit(self.mean_value_function, t, y, p0=p0, method='lm')
        return params, cov

    @abstractmethod
    def mean_value_function(self, t, *args):
        pass