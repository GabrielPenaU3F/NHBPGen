import numpy as np
from scipy.integrate import quad

from domain.processes.gpp import GPP
from exceptions import ModelParametersException


class PenaSigmoidProcess(GPP):

    def __init__(self, gamma, beta, m, l, M, initial_state=0):
        super().__init__(gamma, beta, m, l, M, initial_state=initial_state)

    def rho_t(self, t):
        gamma, beta, m, l, M = self.model_params
        power = ((M - gamma) / (M - m)) ** (t / l)
        return M - (M - m) * power

    def lambda_k_t(self, k, t):
        gamma, beta = self.model_params[:2]
        rho = self.rho_t(t)
        return (beta + gamma * k) / (1 + rho * t)

    def kappa_t(self, t):
        rho = self.rho_t(t)
        return 1/(1 + rho * t)

    def determine_mandatory_parameters(self, *args, **kwargs):
        gamma, beta, m, l, M = args
        return gamma, beta, m, l, M

    def validate_model_parameters(self, model_params):
        gamma, beta, m, l, M = model_params
        super().validate_model_parameters((gamma, beta))
        if not m > 0:
            raise ModelParametersException('Pena Sigmoid m parameter must be a positive number')
        if not l > 0:
            raise ModelParametersException('Pena Sigmoid l parameter must be a positive number')
        if not M > max(gamma, m):
            raise ModelParametersException(r'Pena Sigmoid M parameter must be greater than $\gamma$ and m')
        return gamma, beta, m, l, M

    def generate_next_arrival_time(self, current_state, present_time):

        min_dt = 1e-4
        max_dt = 10.0

        u = np.random.uniform(0, 1)
        k, s = current_state, present_time
        lambda_func = lambda t: self.lambda_k_t(k, t)

        lambda_s = lambda_func(s)
        dt = max(0.1 / (1 + lambda_s), min_dt)
        dt = min(dt, max_dt)

        mid_point = s + dt / 2
        lambda_mid = lambda_func(mid_point)
        lambda_dt = lambda_func(s + dt)

        # Integral por regla de Simpson
        integral_simpson = (dt / 6) * (lambda_s + 4 * lambda_mid + lambda_dt)

        # Cálculo de la tasa promedio
        lambda_avg = integral_simpson / dt if integral_simpson > 0 else lambda_s

        # Generar el tiempo inter-arribo
        delta_t = -np.log(u) / max(lambda_avg, 1e-10)  # Evitar divisiones por cero

        # Actualizar el tiempo
        return present_time + delta_t

    # def generate_next_arrival_time(self, current_state, present_time):
    #
    #     u = np.random.uniform(0, 1)
    #     k, s = current_state, present_time
    #     lambda_func = lambda t: self.lambda_k_t(k, t)
    #     lambda_s = lambda_func(s)
    #     dt = 0.1 / (1 + 10 * lambda_s) # This may be adjusted
    #     mid_point = s + dt / 2
    #     lambda_mid = lambda_func(mid_point)
    #     lambda_dt = lambda_func(s + dt)
    #     integral_trap = (lambda_s + lambda_dt) / 2
    #     integral_simpson = (dt / 6) * (lambda_s + 4 * lambda_mid + lambda_dt)
    #
    #     lambda_avg = integral_simpson / dt
    #     delta_t = -np.log(u) / lambda_avg
    #     return present_time + delta_t
