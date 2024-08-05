import numpy as np

from domain.averagers.time_averager import TimeAverager


class EnsembleTimeAverager:

    def average(self, model, N, T, time_step, average_type='regular'):
        ensemble = self.generate_ensemble(model, N, T, time_step, average_type)
        return np.mean(ensemble)

    # Time: total time to simulate
    # Time step: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, model, N, T, delta, time_step):
        ensemble = self.generate_tamsd_ensemble(model, N, T, delta, time_step)
        return np.mean(ensemble)

    def generate_ensemble(self, model, N, T, time_step, average_type):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            average = time_averager.average(model, T, time_step, average_type)
            ensemble.append(average)
        return ensemble

    def generate_tamsd_ensemble(self, model, N, T, delta, time_step):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            tamsd = time_averager.tamsd(model, T, delta, time_step)
            ensemble.append(tamsd)
        return ensemble


