import numpy as np
from matplotlib import pyplot as plt

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.data_management.data_manager import DataManager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 2000
delta = 100
min_delta = 10
time_step = 1

data = DataManager.load_data('data\\120_agg_1d.csv')
ensemble = np.load('..\\..\\data\\simulation_data\\cesnet_120_1d_ensemble_icdf_sim.npz')['ensemble']
t = data['t_idx'][:]
t2 = np.arange(0, 300)
y = data['cumul_packets'][:]
plt.plot(t, y)
plt.plot(t2, np.mean(ensemble, axis=0))
plt.show()

# H = ea.estimate_hurst(ensemble, T, time_step)
# M = eta.estimate_moses(ensemble, T, delta, time_step)
# L = eta.estimate_noah(ensemble, M, T, delta, time_step)
# J = eta.estimate_joseph(ensemble, min_delta, delta, time_step)
#
# summation = M+L+J-1
# print(f'Summation: {summation}')
