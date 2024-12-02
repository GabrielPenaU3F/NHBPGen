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

    def lambda_k_t(self, k, t):
        gamma, beta = self.model_params[:2]
        rho = self.rho_t(t)
        return (beta + gamma * k) / (1 + rho * t)

    def kappa_t(self, t):
        rho = self.rho_t(t)
        return 1/(1 + rho * t)

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
        gamma, beta, l, M = args
        return gamma, beta, l, M

    def validate_model_parameters(self, model_params):
        gamma, beta, l, M = model_params
        super().validate_model_parameters((gamma, beta))
        if not l > 0:
            raise ModelParametersException('Pena Sigmoid l parameter must be a positive number')
        if not M > 0:
            raise ModelParametersException('Pena Sigmoid M parameter must be a positive number')
        return gamma, beta, l, M

    def generate_next_arrival_time(self, current_state, present_time):

        method_flag = 'simpson'  # Change this to try different averaging methods

        u = np.random.uniform(0, 1)
        k, s = current_state, present_time
        lambda_func = lambda t: self.lambda_k_t(k, t)
        lambda_s = lambda_func(s)
        dt = 0.1 / (1 + 10 * lambda_s) # This may be adjusted
        lambda_dt = lambda_func(s + dt)

        if method_flag == 'linear':
            lambda_avg = (lambda_s + lambda_dt) / 2

        elif method_flag == 'simpson':
            mid_point = s + dt / 2
            lambda_mid = lambda_func(mid_point)
            integral = (dt / 6) * (lambda_s + 4 * lambda_mid + lambda_dt)
            lambda_avg = integral / dt


        elif method_flag == 'quad':
            lambda_avg = quad(lambda_func, s, s + dt)[0] / dt

        delta_t = -np.log(u) / lambda_avg
        return present_time + delta_t


    def interarrival_inverse_cdf(self, current_state, present_time):
        pass
