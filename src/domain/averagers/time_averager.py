import numpy as np

from domain.extra_functions import extra_functions


class TimeAverager:

    # Window length is measured in samples, i.e., multiples of step_length
    def average(self, model, T, window_length, step_length=1):
        states = self.create_normalized_sample_path(model, T, step_length)
        chunked_array = np.array(extra_functions.chunk_array(states, window_length))
        return np.mean(chunked_array, axis=0)

    # delta should be a multiple of step length. Step length is recommended to be 1 to avoid issues
    def tamsd(self, model, T, delta, step_length=1):
        delta = int(delta)  # Just to avoid numpy type issues
        X = self.create_normalized_sample_path(model, T, step_length)
        N = T * step_length
        m = int(delta/step_length)
        displacements = []

        for k in range(0, N - m):
            displacement = (X[k + delta] - X[k]) ** 2
            displacements.append(displacement)

        tamsd = np.mean(displacements)
        return tamsd

    def create_normalized_sample_path(self, model, T, step_length):
        arrivals = model.generate_sample_path(T, plot=False)
        number_of_steps = int(np.floor(T / step_length))
        return extra_functions.generate_equally_spaced_observations(arrivals, step_length, number_of_steps)
