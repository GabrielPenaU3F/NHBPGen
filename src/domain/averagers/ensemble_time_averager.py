import numpy as np

from domain.averagers.time_averager import TimeAverager
from domain.sampler import Sampler


class EnsembleTimeAverager:

    # See e.g. Eq. B1 from Aghion et al. (2021) or Eq. 4 from Vilk et al. (2022)
    def average(self, model, N, T, time_step, average_type='regular'):
        ensemble = self.generate_ensemble(model, N, T, time_step, average_type)
        return np.mean(ensemble)

    # See e.g. Eqs. 4 and 5 from Aghion et al. (2021)
    # min_T: minimum time location over which to compute the average of displacements
    # max_T: maximum of those
    def average_as_function_of_t(self, model, N, min_T, max_T, time_step, average_type='regular'):
        ensemble = self.generate_ensemble_as_function_of_t(model, N, min_T, max_T, time_step, average_type)
        return np.mean(ensemble, axis=0)

    # Time: total time to simulate
    # Time step: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, model, N, T, delta_min, max_delta, time_step):
        ensemble = self.generate_tamsd_ensemble(model, N, T, delta_min, max_delta, time_step)
        return np.mean(ensemble, axis=0)

    def generate_ensemble(self, model, N, T, time_step, average_type):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            observations = Sampler().generate_observations_sample_path(model, T, time_step, plot=False)
            average = time_averager.average(observations, T, time_step, average_type)
            ensemble.append(average)
        return ensemble

    def generate_tamsd_ensemble(self, model, N, T, min_delta, max_delta, time_step):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            tamsd = time_averager.tamsd(model, T, min_delta, max_delta, time_step)
            ensemble.append(tamsd)
            print(f"Generating trajectory n={i + 1} ...")
        return ensemble

    def generate_ensemble_as_function_of_t(self, model, N, min_T, max_T, time_step, average_type):
        time_averager = TimeAverager()
        ensemble = []
        t_axis = np.arange(min_T, max_T, time_step)
        for i in range(N):
            avgs = []
            observations_sample_path = Sampler().generate_observations_sample_path(model, max_T, time_step, plot=False)
            for T in t_axis:
                avg = time_averager.average(observations_sample_path, T, time_step, average_type)
                avgs.append(avg)
            ensemble.append(avgs)
            print(f"Generating trajectory n={i+1} ...")
        return np.array(ensemble)
