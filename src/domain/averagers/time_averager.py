import numpy as np

from domain.extra_functions import extra_functions


class TimeAverager:

    # Window length is measured in samples, i.e., multiples of step_length
    def average(self, model, T, window_length, step_length=1):
        arrivals = model.generate_sample_path(T, plot=False)
        number_of_steps = int(np.floor(T / step_length))
        states = extra_functions.generate_equally_spaced_observations(arrivals, step_length, number_of_steps)
        chunked_array = np.array(extra_functions.chunk_array(states, window_length))
        return np.mean(chunked_array, axis=0)
