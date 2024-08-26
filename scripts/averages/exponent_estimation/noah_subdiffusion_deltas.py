import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

averager = EnsembleTimeAverager()
N = 100
min_T = 1000
max_T = 100000
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_subdif.npz')
ensemble = data['ensemble'][0:N-1]

t = np.arange(min_T, max_T, time_step)
avgs_t_1 = averager.average_as_function_of_t(ensemble, min_T, max_T, 100, time_step, average_type='sq')
avgs_t_2 = averager.average_as_function_of_t(ensemble, min_T, max_T, 200, time_step, average_type='sq')
avgs_t_5 = averager.average_as_function_of_t(ensemble, min_T, max_T, 500, time_step, average_type='sq')
avgs_t_10 = averager.average_as_function_of_t(ensemble, min_T, max_T, 1000, time_step, average_type='sq')

np.savez_compressed('../../../simulation_data/noah_subdiff_averages_for_different_deltas.npz',
                    delta1=avgs_t_1, delta2=avgs_t_2, delta5=avgs_t_5, delta10=avgs_t_10)

# avgs_delta_data = np.load('../../../simulation_data/noah_subdiff_averages_for_different_deltas.npz')
# avgs_t_1 = avgs_delta_data['delta1']
# avgs_t_2 = avgs_delta_data['delta2']
# avgs_t_5 = avgs_delta_data['delta5']
# avgs_t_10 = avgs_delta_data['delta10']
#
# slope1, intercept1, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_1)[50:])
# slope2, intercept2, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_2)[50:])
# slope5, intercept5, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_5)[50:])
# slope10, intercept10, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_10)[50:])
#
# M1 = -0.1689
# M2 = -0.1535
# M5 = -0.1128
# M10 = -0.0658
# L1 = (slope1 - 2 * M1 + 2) / 2
# L2 = (slope2 - 2 * M2 + 2) / 2
# L5 = (slope5 - 2 * M5 + 2) / 2
# L10 = (slope10 - 2 * M10 + 2) / 2
# print(f'L1={L1}')
# print(f'L2={L2}')
# print(f'L5={L5}')
# print(f'L10={L10}')
#
# fit_t = np.linspace(min(t), max(t), 1000)
# fit_1 = 10**(intercept1 + slope1 * np.log10(fit_t)[50:])
# fit_2 = 10**(intercept2 + slope2 * np.log10(fit_t)[50:])
# fit_5 = 10**(intercept5 + slope5 * np.log10(fit_t)[50:])
# fit_10 = 10**(intercept10 + slope10 * np.log10(fit_t)[50:])
#
# plt.loglog(t[50:], avgs_t_1[50:], label='Δ=1')
# plt.loglog(fit_t[50:], fit_1, linestyle='--', label='Scaling law Δ=1')
# plt.loglog(t[50:], avgs_t_2[50:], label='Δ=2')
# plt.loglog(fit_t[50:], fit_2, linestyle='--',  label='Scaling law Δ=2')
# plt.loglog(t[50:], avgs_t_5[50:], label='Δ=5')
# plt.loglog(fit_t[50:], fit_5, linestyle='--',  label='Scaling law Δ=5')
# plt.loglog(t[50:], avgs_t_10[50:], label='Δ=10')
# plt.loglog(fit_t[50:], fit_10, linestyle='--',  label='Scaling law Δ=10')
#
# plt.legend()
# plt.show()