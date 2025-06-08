import numpy as np

from domain.fitting.fitter_models.fitter_model import FitterModel


class BPM3pFitterModel(FitterModel):

    def mean_value_function(self, t, *args, ):
        gamma, beta, rho, = args
        r = beta/gamma
        p = np.exp(-gamma * self.Kappa_function(0, t, rho))
        return self.initial_state + r * (1 - p) / p

    def Kappa_function(self, s, t, rho):
        return (1 / rho) * np.log((1 + rho * t) / (1 + rho * s))
