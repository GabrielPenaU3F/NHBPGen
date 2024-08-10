import warnings

import numpy as np

from domain.averagers.averaging_strategies.time_averaging_strategies import \
    AbsoluteVelocityTimeAveragingStrategy, SquareVelocityTimeAveragingStrategy
from domain.sampler import Sampler


class TimeAverager:

    strategies = {
        'abs-vel': AbsoluteVelocityTimeAveragingStrategy,
        'sq-vel': SquareVelocityTimeAveragingStrategy
    }

    def average(self, observations_sample_path, T, delta, average_type='regular'):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        return strategy_class().calculate(observations_sample_path, T, delta)

    def tamsd(self, model, T, min_delta, max_delta, time_step=1):
        delta_axis = np.arange(min_delta, max_delta + 1)
        tamsd_delta = []
        for delta in delta_axis:
            tamsd = self.calculate_tamsd(model, T, delta, time_step)
            if tamsd == 0:
                tamsd = np.min(tamsd_delta)
            tamsd_delta.append(tamsd)
        return tamsd_delta

    def calculate_tamsd(self, model, T, delta, time_step):
        delta = int(delta)  # Ensure it is an integer
        X = Sampler().generate_observations_sample_path(model, T, time_step, plot=False)
        N = int(T * time_step)
        m = int(delta / time_step)
        displacements = []
        if m >= N:
            m -= 1
            warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")

        for k in range(0, N - m):
            displacement = (X[k + delta] - X[k]) ** 2
            displacements.append(displacement)
        return np.sum(displacements) / (N - m + 1)
