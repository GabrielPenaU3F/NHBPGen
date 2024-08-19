import numpy as np
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.sampler.sampler import Sampler


class EnsembleTimeAverager:

    def average(self, model, N, T, time_step, average_type='regular'):
        ensemble = self.generate_ensemble(model, N, T, time_step, average_type)
        return np.mean(ensemble)

    # See e.g. Eqs. 4 and 5 from Aghion et al. (2021)
    # min_T: minimum time location over which to compute the average of displacements
    # max_T: maximum of those
    def average_as_function_of_t(self, model, N, min_T, max_T, time_step, average_type='regular'):
        ensemble = self.generate_ensemble_as_function_of_t(model, N, min_T, max_T, time_step, average_type)
        return np.mean(ensemble, axis=0)

    # See e.g. Eq. B1 from Aghion et al. (2021) or Eq. 4 from Vilk et al. (2022)
    # Time: total time to simulate
    # Time step: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, model, N, T, min_delta, max_delta, time_step):
        ensemble = self.generate_tamsd_ensemble(model, N, T, min_delta, max_delta, time_step)
        return np.mean(ensemble, axis=0)

    def generate_ensemble(self, model, N, T, time_step, average_type):
        time_averager = TimeAverager()
        ensemble = []
        for i in range(N):
            observations = Sampler().simulate_sample_path(model, T, path_type='observations',
                                                          time_step=time_step, plot=False)
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
            observations_sample_path = Sampler().simulate_sample_path(model, max_T, path_type='observations',
                                                                      time_step=time_step, plot=False)
            for T in t_axis:
                avg = time_averager.average(observations_sample_path, T, time_step, average_type)
                avgs.append(avg)
            ensemble.append(avgs)
            print(f"Generating trajectory n={i+1} ...")
        return np.array(ensemble)

    def estimate_moses(self, model, N, min_T, max_T, time_step=1):
        t = np.arange(min_T, max_T, time_step)
        vel_avgs = self.average_as_function_of_t(model, N, min_T, max_T, time_step, average_type='abs-vel')
        slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(vel_avgs))
        M = slope + 1/2
        print(f'Moses exponent: M={M}')
        return M

    def estimate_noah(self, model, moses, N, min_T, max_T, time_step=1):
        t = np.arange(min_T, max_T, time_step)
        sq_vel_avgs = self.average_as_function_of_t(model, N, min_T, max_T, time_step, average_type='sq-vel')
        slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(sq_vel_avgs))
        L = (slope - 2*moses + 2)/2
        print(f'Noah exponent: L={L}')
        return L

    def estimate_joseph(self, model, N, T, min_delta, max_delta, time_step=1):
        etamsd = self.etamsd(model, N, T, min_delta, max_delta, time_step)
        delta_axis = np.arange(min_delta, max_delta + 1, 1)
        slope, intercept, r_value, p_value, std_err = linregress(np.log(delta_axis), np.log(etamsd))
        J = slope/2
        print(f'Joseph exponent: J={J}')
        return J
