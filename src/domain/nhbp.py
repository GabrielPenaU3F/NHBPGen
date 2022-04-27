import numbers
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from exceptions import ModelParametersException


class NHBP(ABC):

    n_of_params = None
    model_params = None
    initial_state = None

    def __init__(self, model_params, initial_state):
        self.initial_state = initial_state
        self.set_number_of_parameters()
        self.check_number_of_parameters(model_params)
        self.model_params = model_params

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    @abstractmethod
    def intensity_function(self, r, t):
        pass

    def get_initial_state(self):
        return self.initial_state

    def check_number_of_parameters(self, model_params):
        if isinstance(model_params, numbers.Number):
            if 1 != self.n_of_params:
                raise ModelParametersException('Incorrect number of parameters for this model')
        elif len(model_params) != self.n_of_params:
            raise ModelParametersException('Incorrect number of parameters for this model')

    def generate_arrivals(self, time, show=False):
        arrivals = []
        present_time = 0
        current_state = self.initial_state
        while present_time < time:
            current_lambda = self.intensity_function(current_state, present_time)
            interarrival_time = np.random.exponential(scale=1/current_lambda)
            current_state += 1
            present_time += interarrival_time
            if present_time > time:
                break
            arrivals.append(present_time)

        if show is True:
            ys = np.zeros(len(arrivals))
            fig, axes = plt.subplots(figsize=(8, 2))
            axes.scatter(arrivals, ys, marker='o')
            axes.set_xticks(arrivals)
            axes.set_yticks([])
            plt.show()

        return np.array(arrivals)

    def generate_sample_path(self, time, fs):
        pass
