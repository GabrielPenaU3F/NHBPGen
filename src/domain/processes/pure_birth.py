from abc import abstractmethod, ABC

import numpy as np

from domain.processes.nhbp import NHBP


class PureBirthProcess(NHBP, ABC):

    def generate_next_arrival_time(self, current_state, present_time):
        updated_lambda = self.intensity_function(current_state, -1)
        scale = 1/updated_lambda
        return np.random.exponential(scale)
