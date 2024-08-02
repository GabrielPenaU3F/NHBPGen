import numpy as np

from domain.averagers.time_averager import TimeAverager


class EnsembleTimeAverager:

    # Time: total time to simulate
    # Step length: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, model, N, T, delta, step_length):
        ensemble = self.generate_ensemble(model, N, T, delta, step_length)
        average = np.mean(ensemble)
        return average

    def generate_ensemble(self, model, N, T, delta, step_length):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            tamsd = time_averager.tamsd(model, T, delta, step_length)
            ensemble.append(tamsd)
        return ensemble
    
