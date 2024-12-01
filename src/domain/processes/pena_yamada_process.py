import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class PenaYamadaProcess(GPP):

    def __init__(self, gamma, beta, b, initial_state=0):
        super().__init__(gamma, beta, b, initial_state=initial_state)

    def kappa_t(self, t):
        gamma, _, b = self.model_params
        frac = b * t / (2 * np.exp(b * t) - b * t - 1)
        return (b / gamma) * frac

    def Kappa_t(self, t):
        return self.Kappa_s_t(0, t)

    def Kappa_s_t(self, s, t):
        integrando = lambda x: self.kappa_t(x)
        if np.isscalar(t):  # if t is a scalar
            return quad(integrando, s, t)[0]
        else:  # if t is an array
            result = np.zeros_like(t, dtype=float)
            for i, ti in enumerate(t):
                result[i] = quad(integrando, s, ti)[0]
            return result

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, b = args
        return gamma, beta, b

    def validate_model_parameters(self, model_params):
        gamma, beta, b = model_params
        super().validate_model_parameters((gamma, beta))
        if not b > 0:
            raise ModelParametersException('Pena-Yamada b parameter must be a positive number')
        return gamma, beta, b

    def generate_next_arrival_time(self, current_state, present_time):
        k, s = current_state, present_time
        t_values = np.linspace(s, s + 10, 100)  # Adjust this eventually
        F_values = np.array([self.F_t(k, s, t) for t in t_values])
        F_inverse = interp1d(F_values, t_values, bounds_error=False, fill_value="extrapolate")
        u = np.random.uniform(0, 1)
        return F_inverse(u).item()

    def interarrival_inverse_cdf(self, current_state, present_time):
        pass

