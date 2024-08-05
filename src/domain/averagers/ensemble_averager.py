import numpy as np

from domain.extra_functions.extra_functions import create_normalized_sample_path


class EnsembleAverager:

    # N is the total of simulations to run
    # T is the total time units to simulate
    # Time step is the timestep between sucessive observations, in time units. Default is 1
    def average(self, model, N, T, time_step=1):
        ensemble = self.generate_ensemble(model, N, T, time_step)
        average = np.mean(ensemble, axis=0)
        return average.tolist()

    def generate_ensemble(self, model, N, T, time_step):
        ensemble = []
        for i in range(N):
            sample_path = create_normalized_sample_path(model, T, time_step)
            ensemble.append(sample_path)
        return ensemble
