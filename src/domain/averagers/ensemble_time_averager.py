import numpy as np
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager


class EnsembleTimeAverager:

    # See e.g. Eqs. 4 and 5 from Aghion et al. (2021)
    # min_T: minimum time location over which to compute the average of displacements
    # max_T: maximum of those
    def average_as_function_of_t(self, ensemble, min_T, max_T, time_step, average_type='regular'):
        ta_ensemble_t = self.time_average_ensemble_as_function_of_t(ensemble, min_T, max_T, time_step, average_type)
        return np.mean(ta_ensemble_t, axis=0)

    # See e.g. Eq. B1 from Aghion et al. (2021) or Eq. 4 from Vilk et al. (2022)
    # Time: total time to simulate
    # Time step: timestep between computed observations
    # N: Number of repetitions to average
    # Delta: Length of displacement to compute
    def etamsd(self, ensemble, min_delta, max_delta, time_step):
        tamsd_ensemble = self.build_tamsd_ensemble(ensemble, min_delta, max_delta, time_step)
        return np.mean(tamsd_ensemble, axis=0)

    def build_tamsd_ensemble(self, ensemble, min_delta, max_delta, time_step):
        time_averager = TimeAverager()
        tamsd_ensemble = []
        for i in range(len(ensemble)):
            sample_path = ensemble[i]
            tamsd = time_averager.tamsd(sample_path, min_delta, max_delta, time_step)
            tamsd_ensemble.append(tamsd)
            print(f"Time-averaging trajectory n={i + 1} ...")
        return tamsd_ensemble

    def time_average_ensemble_as_function_of_t(self, ensemble, min_T, max_T, time_step, average_type):
        time_averager = TimeAverager()
        avg_ensemble_t = []
        for i in range(len(ensemble)):
            path = ensemble[i]
            avg_t = time_averager.time_average_as_function_of_t(path, min_T, max_T, time_step, average_type)
            avg_ensemble_t.append(avg_t)
            print(f"Time-averaging trajectory n={i+1} ...")
        return np.array(avg_ensemble_t)

    def estimate_moses(self, ensemble, min_T, max_T, time_step=1):
        t = np.arange(min_T, max_T, time_step)
        avgs_t = self.average_as_function_of_t(ensemble, min_T, max_T, time_step, average_type='abs')
        slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(avgs_t))
        M = slope + 1/2
        print(f'Moses exponent: M={M}')
        return M

    def estimate_noah(self, ensemble, moses, min_T, max_T, time_step=1):
        t = np.arange(min_T, max_T, time_step)
        sq_avgs_t = self.average_as_function_of_t(ensemble, min_T, max_T, time_step, average_type='sq')
        slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(sq_avgs_t))
        L = (slope - 2*moses + 2)/2
        print(f'Noah exponent: L={L}')
        return L

    def estimate_joseph(self, ensemble, min_delta, max_delta, time_step=1):
        etamsd = self.etamsd(ensemble, min_delta, max_delta, time_step)
        delta_axis = np.arange(min_delta, max_delta + 1, 1)
        slope, intercept, r_value, p_value, std_err = linregress(np.log(delta_axis), np.log(etamsd))
        J = slope/2
        print(f'Joseph exponent: J={J}')
        return J
