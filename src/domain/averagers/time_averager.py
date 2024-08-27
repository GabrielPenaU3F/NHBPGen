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
            displacements = (sample_path[n_delta:] - sample_path[:-n_delta]) ** 2
            tamsd = np.mean(displacements)
            tamsd_delta.append(tamsd)

        return tamsd_delta

    # def tamsd(self, sample_path, min_delta, max_delta, time_step=1):
    #     delta_axis = np.arange(min_delta, max_delta + 1)
    #     tamsd_delta = []
    #     for delta in delta_axis:
    #         n_delta = int(delta / time_step)
    #         tamsd = self.calculate_tamsd(sample_path, n_delta)
    #         tamsd_delta.append(tamsd)
    #     return tamsd_delta

    # def calculate_tamsd(self, X, n_delta):
    #     N = len(X)
    #     m = n_delta
    #     displacements = []
    #     if m >= N:
    #         m -= 1
    #         warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")
    #
    #     for k in range(0, N - m):
    #         displacement = (X[k + m] - X[k]) ** 2
    #         displacements.append(displacement)
    #     return np.sum(displacements) / (N - m + 1)

    def time_average_as_function_of_t(self, X, min_T, max_T, delta, time_step, average_type):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")
        avg_strategy = strategy_class()

        results = [0]

        # Calculation of the average up to min_T
        n_min = int(min_T / time_step)
        n_delta = int(delta / time_step)
        if n_delta > int(min_T / time_step):
            raise IndexError('The delta time gap must be smaller than min_T')

        sum_increments = 0
        for j in range(n_delta, n_min):
            n1 = j * n_delta
            n2 = (j - 1) * n_delta
            if n1 < len(X):
                sum_increments += avg_strategy.calculate(X, n1, n2)
            else:
                break
        results.append(sum_increments / (n_min - n_delta + 1))

        # Calculation of all subsequent averages by reusing the cumulative sum
        T_axis = np.arange(min_T + time_step, max_T, time_step)
        for T in T_axis:
            n = int(T / time_step)
            if n < len(X):
                sum_increments += avg_strategy.calculate(X, n, n - n_delta)
                results.append(sum_increments / (n - n_delta + 1))
            else:
                break

        return results[1:]
