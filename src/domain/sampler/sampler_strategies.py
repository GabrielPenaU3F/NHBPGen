from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from domain.extra_functions import extra_functions
from domain.processes.nhbp import NHBP
from exceptions import SimulationException


class SamplePathStrategy(ABC):

    @abstractmethod
    def generate_path(self, *args: Any, **kwargs: Any) -> np.typing.NDArray:
        pass

    @abstractmethod
    def generate_ensemble(self, *args, **kwargs) -> np.typing.NDArray:
        pass


class ArrivalsSamplePathStrategy(SamplePathStrategy):

    def generate_path(self, model: NHBP, time: float, plot=True) -> np.typing.NDArray:
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

    def generate_ensemble(self, *args, **kwargs) -> np.typing.NDArray:
        pass


class ObservationsSamplePathStrategy(SamplePathStrategy):

    def generate_path(self, model: NHBP, time: float, time_step: float, plot=True, dtype='int64') -> np.typing.NDArray:
        arrivals = model.generate_arrivals(time)
        time_markers = np.arange(0, time, time_step, dtype=dtype)
        observations = extra_functions.parse_observations(arrivals, time_markers, dtype=dtype)

        if plot is True:
            t = np.arange(0, time, time_step)
            fig, axes = plt.subplots(figsize=(12, 5))
            axes.step(t, observations, where='post')
            plt.show()

        return observations

    def generate_ensemble(self, model: NHBP, N: int, time: float, time_step: float, plot=True, dtype='int64') -> np.typing.NDArray:
        ensemble = []
        time_markers = np.arange(0, time, time_step, dtype=dtype)
        for i in range(N):
            arrivals = model.generate_arrivals(time)
            sample_path = extra_functions.parse_observations(arrivals, time_markers)
            ensemble.append(sample_path)
            print(f"Generating trajectory n={i + 1} ...")
        return np.array(ensemble)
