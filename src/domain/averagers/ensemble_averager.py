import numpy as np

from domain.extra_functions import extra_functions


class EnsembleAverager:

    # N is the total of simulations to run
    # T is the totsl time units to simulate
    # step_length is how often we look at the values, in time units. Default is 1
    def average(self, model, N, T, step_length=1):
        arrival_times_ensemble = self.generate_ensemble(model, N, T)
        states_ensemble = []
        number_of_steps = int(np.floor(T/step_length))
        for sample_path in arrival_times_ensemble:
            states = extra_functions.generate_equally_spaced_observations(sample_path, step_length, number_of_steps)
            states_ensemble.append(states)

        ensemble_array = np.array(states_ensemble)
        average = np.mean(ensemble_array, axis=0)
        return average.tolist()

    def generate_ensemble(self, model, N, T):
        ensemble = []
        for i in range(N):
            arrivals = model.generate_sample_path(T, plot=False)
            ensemble.append(arrivals)
        return ensemble
