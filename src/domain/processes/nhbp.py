from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from exceptions import ModelParametersException


class NHBP(ABC):

    model_params = None
    initial_state = None

    def __init__(self, *args, **kwargs):
        self.initial_state = self.validate_initial_state(kwargs.get('initial_state'))
        mandatory_params = self.determine_mandatory_parameters(*args)
        self.model_params = self.validate_model_parameters(mandatory_params)

    @abstractmethod
    def intensity_function(self, k, t):
        pass

    def get_initial_state(self):
        return self.initial_state

    def validate_initial_state(self, initial_state):
        if not initial_state:
            return 0
        if not isinstance(initial_state, int) or initial_state < 0:
            raise ModelParametersException('Initial state must be a positive integer')
        return initial_state

    @abstractmethod
    def determine_mandatory_parameters(self, *args, **kwargs):
        pass

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

        return np.array(arrivals, dtype='float32')

    @abstractmethod
    def generate_next_arrival_time(self, current_state, present_time):
        pass

    def get_model_parameters(self):
        return self.model_params

