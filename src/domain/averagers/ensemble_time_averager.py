import numpy as np

from domain.averagers.time_averager import TimeAverager


class EnsembleTimeAverager:

    def ensemble_time_absolute_velocity_average(self, model, N, T, window_length, time_step):
        ensemble = self.generate_absolute_velocity_ensemble(model, N, T, window_length, time_step)
        return np.mean(ensemble, axis=0)

    # Time: total time to simulate
    # Time step: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, model, N, T, delta, time_step):
        ensemble = self.generate_tamsd_ensemble(model, N, T, delta, time_step)
        return np.mean(ensemble)

    def generate_absolute_velocity_ensemble(self, model, N, T, window_length, time_step):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            absolute_average = time_averager.absolute_velocity_average(model, T, window_length, time_step)
            ensemble.append(absolute_average)
        return ensemble

    def generate_tamsd_ensemble(self, model, N, T, delta, time_step):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            tamsd = time_averager.tamsd(model, T, delta, time_step)
            ensemble.append(tamsd)
        return ensemble


