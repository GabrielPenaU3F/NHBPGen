import warnings

import numpy as np

from domain.averagers.displacement_averaging_strategies.displacement_averaging_strategies import \
    RegularAveragingStrategy, AbsoluteAveragingStrategy, AbsoluteVelocityAveragingStrategy, SquaredAveragingStrategy
from domain.extra_functions import extra_functions


class TimeAverager:

    strategies = {}

    def __init__(self):
        self.strategies = {
            'regular': RegularAveragingStrategy,
            'abs': AbsoluteAveragingStrategy,
            'abs-vel': AbsoluteVelocityAveragingStrategy,
            'msd': SquaredAveragingStrategy
        }

    # Window length is measured in samples, i.e., multiples of time_step
    def average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step)
        return np.mean(displacements)

    def absolute_average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='abs')
        return np.mean(np.abs(displacements))

    def absolute_velocity_average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='abs-vel')
        return np.mean(np.abs(displacements))

    # delta should be a multiple of time step.
    def tamsd(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='msd')
        return np.mean(displacements)

    def generate_displacements_array(self, model, T, delta, time_step, average_type='regular'):
        delta = int(delta)  # Ensure it is an integer
        X = extra_functions.create_normalized_sample_path(model, T, time_step)
        N = int(T * time_step)
        m = int(delta / time_step)
        displacements = []
        if m >= N:
            m -= 1
            warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")

        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        averaging_strategy = strategy_class()
        for k in range(0, N - m):
            displacement = averaging_strategy.calculate(X, k, delta)
            displacements.append(displacement)
        return displacements
