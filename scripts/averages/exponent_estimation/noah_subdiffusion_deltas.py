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
# ensemble = data['ensemble'][0:N]
# 
# avgs_t_10 = averager.average_as_function_of_t(ensemble, min_T, max_T, 100, time_step, average_type='sq')
# avgs_t_20 = averager.average_as_function_of_t(ensemble, min_T, max_T, 200, time_step, average_type='sq')
# avgs_t_50 = averager.average_as_function_of_t(ensemble, min_T, max_T, 500, time_step, average_type='sq')
# avgs_t_100 = averager.average_as_function_of_t(ensemble, min_T, max_T, 1000, time_step, average_type='sq')
# 
# np.savez_compressed('../../../simulation_data/noah_subdiff_averages_for_different_deltas.npz',
#                     delta10=avgs_t_10, delta20=avgs_t_20, delta50=avgs_t_50, delta100=avgs_t_100)

avgs_delta_data = np.load('../../../simulation_data/noah_subdiff_averages_for_different_deltas.npz')
avgs_t_10 = avgs_delta_data['delta10']
avgs_t_20 = avgs_delta_data['delta20']
avgs_t_50 = avgs_delta_data['delta50']
avgs_t_100 = avgs_delta_data['delta100']

slope10, intercept10, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_10)[100:])
slope20, intercept20, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_20)[100:])
slope50, intercept50, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_50)[100:])
slope100, intercept100, r_value, p_value, std_err = linregress(np.log(t)[100:], np.log(avgs_t_100)[100:])

M10 = 0.0042
M20 = -0.0007
M50 = -0.0410
M100 = -0.1494
L10 = (slope10 - 2 * M10 + 2) / 2
L20 = (slope20 - 2 * M20 + 2) / 2
L50 = (slope50 - 2 * M50 + 2) / 2
L100 = (slope100 - 2 * M100 + 2) / 2
print(f'L10={L10}')
print(f'L20={L20}')
print(f'L50={L50}')
print(f'L100={L100}')

fit_10 = 10**(intercept10 + slope10 * np.log10(t)[100:])
fit_20 = 10**(intercept20 + slope20 * np.log10(t)[100:])
fit_50 = 10**(intercept50 + slope50 * np.log10(t)[100:])
fit_100 = 10**(intercept100 + slope100 * np.log10(t)[100:])

plt.loglog(t[100:], avgs_t_10[100:], label='Δ=10')
plt.loglog(t[100:], fit_10, linestyle='--', label='Scaling law Δ=100')
plt.loglog(t[100:], avgs_t_20[100:], label='Δ=20')
plt.loglog(t[100:], fit_20, linestyle='--',  label='Scaling law Δ=200')
plt.loglog(t[100:], avgs_t_50[100:], label='Δ=50')
plt.loglog(t[100:], fit_50, linestyle='--',  label='Scaling law Δ=500')
plt.loglog(t[100:], avgs_t_100[100:], label='Δ=100')
plt.loglog(t[100:], fit_100, linestyle='--',  label='Scaling law Δ=1000')

plt.legend()
plt.show()