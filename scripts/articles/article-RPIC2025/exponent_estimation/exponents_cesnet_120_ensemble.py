import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 30
T = 100 # Actually it's 300 but we'll discard the first 200 to analyse asymptotically
delta = 10
min_delta = 2
time_step = 1

data = np.load('../../../../data/simulation_data/cesnet_120_1d_ensemble_icdf_sim.npz')
ensemble = data['ensemble']

H = ea.estimate_hurst(ensemble[:, 200:])
M = eta.estimate_moses(ensemble[:, 200:], T, delta, time_step)
L = eta.estimate_noah(ensemble[:, 200:], M, T, delta, time_step)
J = eta.estimate_joseph(ensemble[:, 200:], min_delta, delta, time_step)

summation = M+L+J-1

print('--- Results ---')
print(f'Moses: {M}')
print(f'Noah: {L}')
print(f'Joseph: {J}')
print(f'Hurst: {H}')

print(f'Summation: {summation}')