import numpy as np
from scipy.integrate import quad

from domain.fitting.fitter_models.fitter_model import FitterModel


class PenaSigmoid3pFitterModel(FitterModel):

    def mean_value_function(self, t, *args):
        gamma, l, M, = args
        r = 1/gamma
        p = np.exp(-gamma * self.Kappa_function(t, gamma, l, M))
        return r * (1 - p) / p

    def Kappa_function(self, t, gamma, l, M):
        integrand = lambda x: self.kappa_function(x, gamma, l, M)
        Kx = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            Kx[i] = quad(integrand, 0, ti)[0]
        return Kx

    def kappa_function(self, t, gamma, l, M):
        rho = self.rho(t, gamma, l, M)
        return 1 / (1 + rho * t)

    def rho(self, t, gamma, l, M):
        return M * (1 - ((M - gamma) / M) ** (t / l))
