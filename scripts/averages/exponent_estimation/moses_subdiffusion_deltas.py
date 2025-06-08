import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

averager = EnsembleTimeAverager()
N = 1000
min_T = 1000
max_T = 100000
time_step = 1
t = np.arange(min_T, max_T, time_step)

# data = np.load('../../../test/test_data/test_ensemble_subdif.npz')
# ensemble = data['ensemble']
# 
# avgs_t_10 = averager.average_as_function_of_t(ensemble, min_T, max_T, 100, time_step, average_type='abs')
# avgs_t_20 = averager.average_as_function_of_t(ensemble, min_T, max_T, 200, time_step, average_type='abs')
# avgs_t_50 = averager.average_as_function_of_t(ensemble, min_T, max_T, 500, time_step, average_type='abs')
# avgs_t_100 = averager.average_as_function_of_t(ensemble, min_T, max_T, 1000, time_step, average_type='abs')
# 
# np.savez_compressed('../../../simulation_data/moses_subdiff_averages_for_different_deltas.npz',
#                     delta1=avgs_t_10, delta2=avgs_t_20, delta5=avgs_t_50, delta10=avgs_t_100)
# 
avgs_delta_data = np.load('../../../data/simulation_ensembles/moses_subdiff_averages_for_different_deltas.npz')
avgs_t_10 = avgs_delta_data['delta1']
avgs_t_20 = avgs_delta_data['delta2']
avgs_t_50 = avgs_delta_data['delta5']
avgs_t_100 = avgs_delta_data['delta10']

slope10, intercept10, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_10)[100:])
slope20, intercept20, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_20)[100:])
slope50, intercept50, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_50)[100:])
slope100, intercept100, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_100)[100:])
M10 = slope10 + 1 / 2
M20 = slope20 + 1 / 2
M50 = slope50 + 1 / 2
M100 = slope100 + 1 / 2
print(f'M10={M10}')
print(f'M20={M20}')
print(f'M50={M50}')
print(f'M100={M100}')

fit_1 = 10**(intercept10 + slope10 * np.log10(t)[100:])
fit_2 = 10**(intercept20 + slope20 * np.log10(t)[100:])
fit_5 = 10**(intercept50 + slope50 * np.log10(t)[100:])
fit_10 = 10**(intercept100 + slope100 * np.log10(t)[100:])

plt.loglog(t[100:], avgs_t_10[100:], label='Δ=100')
plt.loglog(t[100:], fit_1, linestyle='--', label='Scaling law Δ=100')
plt.loglog(t[100:], avgs_t_20[100:], label='Δ=200')
plt.loglog(t[100:], fit_2, linestyle='--',  label='Scaling law Δ=200')
plt.loglog(t[100:], avgs_t_50[100:], label='Δ=500')
plt.loglog(t[100:], fit_5, linestyle='--',  label='Scaling law Δ=500')
plt.loglog(t[100:], avgs_t_100[100:], label='Δ=1000')
plt.loglog(t[100:], fit_10, linestyle='--',  label='Scaling law Δ=1000')

plt.legend()
plt.show()
