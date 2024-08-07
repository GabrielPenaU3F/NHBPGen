import warnings

import numpy as np

from domain.averagers.displacement_averaging_strategies.displacement_averaging_strategies import \
    RegularAveragingStrategy, AbsoluteAveragingStrategy, AbsoluteVelocityAveragingStrategy
from domain.sampler import Sampler


class TimeAverager:

    strategies = {
        'regular': RegularAveragingStrategy,
        'abs': AbsoluteAveragingStrategy,
        'abs-vel': AbsoluteVelocityAveragingStrategy,
    }

    def average(self, observations_sample_path, T, time_step=1, average_type='regular'):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        return strategy_class().calculate(observations_sample_path, T, time_step)

    def tamsd(self, model, T, min_delta, max_delta, time_step=1):
        delta_axis = np.arange(min_delta, max_delta + 1)
        tamsd_delta = []
        for delta in delta_axis:
            displacements = self.generate_displacements_array(model, T, delta, time_step)
            tamsd = np.mean(displacements)
            if tamsd == 0:
                tamsd = np.min(tamsd_delta)
            tamsd_delta.append(tamsd)
        return tamsd_delta

    def generate_displacements_array(self, model, T, delta, time_step):
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
        return displacements
