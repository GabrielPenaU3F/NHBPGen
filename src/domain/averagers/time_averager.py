import warnings

import numpy as np

from domain.averagers.averaging_strategies.time_averaging_strategies import \
    AbsoluteTimeAveragingStrategy, SquareTimeAveragingStrategy


class TimeAverager:

    strategies = {
        'abs': AbsoluteTimeAveragingStrategy,
        'sq': SquareTimeAveragingStrategy
    }

    def tamsd(self, sample_path, min_delta, max_delta):
        N = len(sample_path)
        delta_axis = np.arange(min_delta, max_delta + 1)
        tamsd_delta = []

        for delta in delta_axis:

            if delta >= N:
                warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")
                delta = N - 1

            # Use vectorized operations to calculate displacements
            displacements = np.square(sample_path[delta:] - sample_path[:-delta])
            tamsd = np.mean(displacements)
            tamsd_delta.append(tamsd)

        return tamsd_delta

    def time_average_as_function_of_t(self, X, delta, average_type):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")
        avg_strategy = strategy_class()

        t_values = np.arange(delta, len(X) + 1, delta)
        avg_values = []

        for t in t_values:
            n_blocks = t // delta
            sum_increments = 0.0

            for j in range(1, n_blocks + 1):
                n1 = j * delta
                n2 = (j - 1) * delta
                if n1 >= len(X):
                    break
                sum_increments += avg_strategy.calculate(X, n1, n2)

            avg = sum_increments / t
            avg_values.append(avg)

        return np.array(avg_values)