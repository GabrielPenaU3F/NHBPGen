import warnings

import numpy as np

from domain.averagers.averaging_strategies.time_averaging_strategies import \
    AbsoluteTimeAveragingStrategy, SquareTimeAveragingStrategy


class TimeAverager:

    strategies = {
        'abs': AbsoluteTimeAveragingStrategy,
        'sq': SquareTimeAveragingStrategy
    }

    def average(self, observations_sample_path, T, delta, time_step, average_type='regular'):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        return strategy_class().calculate(observations_sample_path, T, delta, time_step)

    def tamsd(self, sample_path, min_delta, max_delta, time_step=1):
        delta_axis = np.arange(min_delta, max_delta + 1)
        tamsd_delta = []
        for delta in delta_axis:
            n_delta = int(delta / time_step)
            tamsd = self.calculate_tamsd(sample_path, n_delta)
            tamsd_delta.append(tamsd)
        return tamsd_delta

    def calculate_tamsd(self, X, n_delta):
        N = len(X)
        m = n_delta
        displacements = []
        if m >= N:
            m -= 1
            warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")

        for k in range(0, N - m):
            displacement = (X[k + m] - X[k]) ** 2
            displacements.append(displacement)
        return np.sum(displacements) / (N - m + 1)

    def time_average_as_function_of_t(self, sample_path, min_T, max_T, delta, time_step, average_type):
        avg_t = []
        t_axis = np.arange(min_T, max_T, time_step)
        for T in t_axis:
            avg = self.average(sample_path, T, delta, time_step, average_type)
            avg_t.append(avg)
        return avg_t
