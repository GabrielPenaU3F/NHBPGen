import numpy as np


from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class PenaSigmoidProcess(GPP):

    def __init__(self, gamma, beta, m, l, M, initial_state=0):
        super().__init__(gamma, beta, m, l, M, initial_state=initial_state)

    def lambda_k_t(self, k, t):
        gamma, beta = self.model_params[:2]
        rho = self.rho_t(t)
        return (beta + gamma * k) / (1 + rho * t)

    def kappa_t(self, t):
        rho = self.rho_t(t)
        return 1/(1 + rho * t)

    def rho_t(self, t):
        gamma, beta, m, l, M = self.model_params
        power = ((M - gamma) / (M - m)) ** (t / l)
        return M - (M - m) * power

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, m, l, M = args
        return gamma, beta, m, l, M

    def validate_model_parameters(self, model_params):
        gamma, beta, m, l, M = model_params
        super().validate_model_parameters((gamma, beta))
        if not m >= 0:
            raise ModelParametersException('Pena Sigmoid m parameter must be a positive number or zero')
        if not l > 0:
            raise ModelParametersException('Pena Sigmoid l parameter must be a positive number')
        if not ((M > gamma and gamma > m) or (M < m and m < gamma) or (M < gamma and gamma < m) or (m == gamma or M == gamma)):
            raise ModelParametersException(r'Pena Sigmoid M parameter must fall in one of the permitted regions')
        return gamma, beta, m, l, M

    def generate_next_arrival_time_alt(self, k, s):

        # Buscamos una cota superior de lambda_k(t) en una ventana corta
        window = 1.0
        t_test = np.linspace(s, s + window, 10)
        lambda_vals = self.lambda_k_t(k, t_test)
        lambda_max = np.max(lambda_vals)
        lambda_bar = lambda_max * 1.1  # Cota con margen de seguridad

        t = s
        while True:
            u = np.random.uniform()
            delta_t = -np.log(u) / lambda_bar
            t_candidate = t + delta_t
            lambda_candidate = self.lambda_k_t(k, t_candidate)

            u2 = np.random.uniform()
            if u2 <= lambda_candidate / lambda_bar:
                return t_candidate
            else:
                t = t_candidate  # Avanzamos y seguimos intentando

    def generate_next_arrival_time(self, k, s):

        min_dt = 1e-4
        max_dt = 1.0

        u = np.random.uniform(0, 1)
        lambda_func = lambda t: self.lambda_k_t(k, t)

        lambda_s = lambda_func(s)
        dt = max(0.1 / (1 + lambda_s), min_dt)
        dt = min(dt, max_dt)

        # Trapezius rule
        lambda_dt = lambda_func(s + dt)
        integral = (dt / 2) * (lambda_s + lambda_dt)

        # Simpson rule
        # mid_point = s + dt / 2
        # lambda_mid = lambda_func(mid_point)
        # lambda_dt = lambda_func(s + dt)
        # integral = (dt / 6) * (lambda_s + 4 * lambda_mid + lambda_dt)

        lambda_avg = integral/ dt if integral > 0 else lambda_s
        delta_t = -np.log(u) / max(lambda_avg, 1e-10)
        return s + delta_t
