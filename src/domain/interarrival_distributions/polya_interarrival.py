from scipy.stats import rv_continuous


class PolyaInterarrival(rv_continuous):

    def __init__(self, gamma, beta, k, s, *args, **kwargs):
        self.gamma = gamma
        self.beta = beta
        self.k = k
        self.s = s
        super().__init__(*args, **kwargs)
        # init our cdf and ppf functions
        # self.cdf_func, self.ppf_func = self.create_cdf_ppf()

    def _pdf(self, x, a=0):
        beta_s_plus_1 = 1 + self.beta * self.s
        beta_t_plus_1 = 1 + self.beta * x
        linear_factor = self.beta + self.gamma * self.k
        exponent = 1 + self.k * self.gamma / self.beta
        return (linear_factor / beta_t_plus_1) * ((beta_s_plus_1 / beta_t_plus_1) ** exponent)
