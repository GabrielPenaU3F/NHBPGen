import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1/4, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
N = 1000
min_T = 10
max_T = 1000
delta = 1
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_subdif.npz')
ensemble = data['ensemble']

t = np.arange(min_T, max_T, time_step)
# avgs_t_1 = averager.average_as_function_of_t(ensemble, min_T, max_T, 1, time_step, average_type='abs')
# avgs_t_2 = averager.average_as_function_of_t(ensemble, min_T, max_T, 2, time_step, average_type='abs')
# avgs_t_5 = averager.average_as_function_of_t(ensemble, min_T, max_T, 5, time_step, average_type='abs')
# avgs_t_10 = averager.average_as_function_of_t(ensemble, min_T, max_T, 10, time_step, average_type='abs')
#
# np.savez_compressed('../../../simulation_data/moses_subdiff_averages_for_different_deltas.npz',
#                     delta1=avgs_t_1, delta2=avgs_t_2, delta5=avgs_t_5, delta10=avgs_t_10)

avgs_delta_data = np.load('../../../simulation_data/moses_subdiff_averages_for_different_deltas.npz')
avgs_t_1 = avgs_delta_data['delta1']
avgs_t_2 = avgs_delta_data['delta2']
avgs_t_5 = avgs_delta_data['delta5']
avgs_t_10 = avgs_delta_data['delta10']

slope1, intercept1, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_1)[50:])
slope2, intercept2, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_2)[50:])
slope5, intercept5, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_5)[50:])
slope10, intercept10, r_value, p_value, std_err = linregress(np.log(t)[50:], np.log(avgs_t_10)[50:])
M1 = slope1 + 1 / 2
M2 = slope2 + 1 / 2
M5 = slope5 + 1 / 2
M10 = slope10 + 1 / 2
print(f'M1={M1}')
print(f'M2={M2}')
print(f'M5={M5}')
print(f'M10={M10}')

fit_t = np.linspace(min(t), max(t), 1000)
fit_1 = 10**(intercept1 + slope1 * np.log10(fit_t)[50:])
fit_2 = 10**(intercept2 + slope2 * np.log10(fit_t)[50:])
fit_5 = 10**(intercept5 + slope5 * np.log10(fit_t)[50:])
fit_10 = 10**(intercept10 + slope10 * np.log10(fit_t)[50:])

plt.loglog(t[50:], avgs_t_1[50:], label='Δ=1')
plt.loglog(fit_t[50:], fit_1, linestyle='--', label='Scaling law Δ=1')
plt.loglog(t[50:], avgs_t_2[50:], label='Δ=2')
plt.loglog(fit_t[50:], fit_2, linestyle='--',  label='Scaling law Δ=2')
plt.loglog(t[50:], avgs_t_5[50:], label='Δ=5')
plt.loglog(fit_t[50:], fit_5, linestyle='--',  label='Scaling law Δ=5')
plt.loglog(t[50:], avgs_t_10[50:], label='Δ=10')
plt.loglog(fit_t[50:], fit_10, linestyle='--',  label='Scaling law Δ=10')

plt.legend()
plt.show()
