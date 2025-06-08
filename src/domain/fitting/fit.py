class Fit:

    def __init__(self, x, y, params, cov, rsq):
        self.x_data = x
        self.y_data = y
        self.params = params
        self.cov = cov
        self.rsq = rsq

    def get_x_data(self):
        return self.x_data

    def get_y_data(self):
        return self.y_data

    def get_params(self):
        return self.params

    def get_cov(self):
        return self.cov

    def get_rsq(self):
        return self.rsq
