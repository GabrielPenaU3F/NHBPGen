from abc import ABC, abstractmethod
from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt

from domain.extra_functions import extra_functions
from domain.processes.nhbp import NHBP
from exceptions import SimulationException


class SamplePathStrategy(ABC):
    @abstractmethod
    def generate_path(self, *args: Any, **kwargs: Any) -> List[float]:
        pass

    @abstractmethod
    def generate_ensemble(self, *args, **kwargs) -> List[List[float]]:
        pass


class ArrivalsSamplePathStrategy(SamplePathStrategy):

    def generate_path(self, model: NHBP, time: float, plot=True) -> List[float]:
        if not time > 0:
            raise SimulationException('Duration of the simulation must be a positive number')
        arrivals = model.generate_arrivals(time)
        if plot is True:
            x_times = np.concatenate((np.array([0]), arrivals, [time]))
            fig, axes = plt.subplots(figsize=(12, 5))
            steps = np.arange(model.get_initial_state(), model.get_initial_state() + len(arrivals) + 1)
            steps = np.append(steps, steps[-1])
            axes.step(x_times, steps, where='post')
            plt.show()
        return arrivals

    def generate_ensemble(self, *args, **kwargs) -> List[List[float]]:
        pass


class ObservationsSamplePathStrategy(SamplePathStrategy):

    def generate_path(self, model: NHBP, time: float, time_step: float, plot=True) -> List[float]:
        arrivals = model.generate_arrivals(time)
        number_of_steps = int(np.floor(time / time_step))
        observations = []
        for i in range(0, number_of_steps):
            time_marker = i * time_step
            observations.append(extra_functions.count_events_until_time(arrivals, time_marker))

        if plot is True:
            t = np.arange(0, time, time_step)
            fig, axes = plt.subplots(figsize=(12, 5))
            axes.step(t, observations, where='post')
            plt.show()

        return observations

    def generate_ensemble(self, *args, **kwargs) -> List[List[float]]:
        pass
