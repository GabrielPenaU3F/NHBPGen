import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class PenaSigmoidProcess(GPP):

    def __init__(self, gamma, beta, l, M, initial_state=0):
        super().__init__(gamma, beta, l, M, initial_state=initial_state)

    def rho_t(self, t):
        gamma, beta, l, M = self.model_params
        power = ((M - 2) / (M - 1)) ** (t / l)
        return M - (M - 1) * power

    def kappa_t(self, t):
        rho = self.rho_t(t)
        return 1/(1 + rho * t)

    def Kappa_t(self, t):
        return self.Kappa_s_t(0, t)

    def Kappa_s_t(self, s, t):
        if np.isscalar(t):  # Si t es un escalar
            integrando = lambda x: self.kappa_t(x)
            return quad(integrando, s, t)[0]
        else:  # Si t es un array
            result = np.zeros_like(t, dtype=float)
            integrando = lambda x: self.kappa_t(x)
            for i, ti in enumerate(t):
                result[i] = quad(integrando, s, ti)[0]
            return result

    def F_t(self, k, s, t):
        gamma, beta = self.model_params[:2]
        integral = self.Kappa_s_t(s, t)
        return 1 - np.exp(-(beta + gamma * k) * integral)

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, l, M = args
        return gamma, beta, l, M

    def validate_model_parameters(self, model_params):
        gamma, beta, l, M = model_params
        if not gamma > 0:
            raise ModelParametersException('Pena Sigmoid gamma parameter must be a positive number')
        if not beta > 0:
            raise ModelParametersException('Pena Sigmoid beta parameter must be a positive number')
        if not l > 0:
            raise ModelParametersException('Pena Sigmoid l parameter must be a positive number')
        if not M > 0:
            raise ModelParametersException('Pena Sigmoid M parameter must be a positive number')
        return gamma, beta, l, M

    def generate_next_arrival_time(self, current_state, present_time):
        k, s = current_state, present_time
        t_values = np.linspace(s, s + 10, 100)  # Adjust this eventually
        F_values = np.array([self.F_t(k, s, t) for t in t_values])
        F_inverse = interp1d(F_values, t_values, bounds_error=False, fill_value="extrapolate")
        u = np.random.uniform(0, 1)
        return F_inverse(u).item()

    def interarrival_inverse_cdf(self, current_state, present_time):
        pass
