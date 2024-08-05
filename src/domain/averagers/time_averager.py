import warnings

import numpy as np

from domain.extra_functions import extra_functions


class TimeAverager:

    # Window length is measured in samples, i.e., multiples of time_step
    def average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step)
        return np.mean(displacements)

    def absolute_average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='abs')
        return np.mean(np.abs(displacements))

    def absolute_velocity_average(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='abs_vel')
        return np.mean(np.abs(displacements))

    # delta should be a multiple of time step.
    def tamsd(self, model, T, delta, time_step=1):
        displacements = self.generate_displacements_array(model, T, delta, time_step, average_type='msd')
        return np.mean(displacements)

    def generate_displacements_array(self, model, T, delta, time_step, average_type=None):
        delta = int(delta)  # Ensure it is an integer
        X = extra_functions.create_normalized_sample_path(model, T, time_step)
        N = int(T * time_step)
        m = int(delta / time_step)
        displacements = []
        if m >= N:
            m -= 1
            warnings.warn("Delta is too large for the given T and step_length. Truncating the last observation.")
        for k in range(0, N - m):
            if average_type == 'msd':
                displacement = (X[k + delta] - X[k]) ** 2
            elif average_type == 'abs':
                displacement = np.abs(X[k + delta] - X[k])
            elif average_type == 'abs_vel':
                displacement = np.abs((X[k + delta] - X[k])/delta)
            else:
                displacement = X[k + delta] - X[k]
            displacements.append(displacement)
        return displacements
