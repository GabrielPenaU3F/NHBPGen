import numbers
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from exceptions import ModelParametersException, SimulationException


class NHBP(ABC):

    n_of_params = None
    model_params = None
    initial_state = None

    def __init__(self, model_params, initial_state=0):
        self.initial_state = self.validate_initial_state(initial_state)
        self.set_number_of_parameters()
        self.check_number_of_parameters(model_params)
        self.model_params = self.validate_model_parameters(model_params)

    @abstractmethod
    def set_number_of_parameters(self):
        pass

    @abstractmethod
    def intensity_function(self, k, t):
        pass

    def get_initial_state(self):
        return self.initial_state

    def check_number_of_parameters(self, model_params):
        if isinstance(model_params, numbers.Number):
            if 1 != self.n_of_params:
                raise ModelParametersException('Incorrect number of parameters for this model')
        elif len(model_params) != self.n_of_params:
            raise ModelParametersException('Incorrect number of parameters for this model')

    def validate_initial_state(self, initial_state):
        if not initial_state >= 0 or not isinstance(initial_state, int):
            raise ModelParametersException('Initial state must be a positive integer')
        return initial_state

    @abstractmethod
    def validate_model_parameters(self, model_params):
        pass

    def generate_arrivals(self, time, show=False):
        arrivals = []
        present_time = 0
        current_state = self.initial_state
        while present_time < time:
            present_time = self.generate_next_arrival_time(current_state, present_time)
            current_state += 1
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

    @abstractmethod
    def generate_next_arrival_time(self, current_state, present_time):
        pass

    def get_model_parameters(self):
        return self.model_params

