import warnings

import numpy as np

from domain.averagers.averaging_strategies.time_averaging_strategies import \
    AbsoluteTimeAveragingStrategy, SquareTimeAveragingStrategy


class TimeAverager:

    strategies = {
        'abs': AbsoluteTimeAveragingStrategy,
        'sq': SquareTimeAveragingStrategy
    }

    def tamsd(self, sample_path, min_delta, max_delta, time_step=1):
        N = len(sample_path)
        delta_axis = np.arange(min_delta, max_delta + 1)
        tamsd_delta = []

        for delta in delta_axis:
            n_delta = int(delta / time_step)

            if n_delta >= N:
                warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")
                n_delta = N - 1

            # Use vectorized operations to calculate displacements
            displacements = np.square(sample_path[n_delta:] - sample_path[:-n_delta])
            tamsd = np.mean(displacements)
            tamsd_delta.append(tamsd)

        return tamsd_delta

    def time_average_as_function_of_t(self, X, max_T, delta, time_step, average_type):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")
        avg_strategy = strategy_class()

        results = [0]
        n_delta = int(delta / time_step)
        sum_increments = 0
        if n_delta < len(X):
            sum_increments += avg_strategy.calculate(X, n_delta, 0)
        results.append(sum_increments / (n_delta + 1))

        # Calculation of all subsequent averages by reusing the cumulative sum
        T_axis = np.arange(delta, max_T, time_step)
        for T in T_axis:
            n = int(T / time_step)
            if n < len(X):
                sum_increments += avg_strategy.calculate(X, n, n - n_delta)
                results.append(sum_increments / (n - n_delta + 1))
            else:
                break

        return results[1:]
