import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 10000
min_T = 10
max_T = 1000
delta = 1
min_delta = 10
max_delta = 100
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_normaldif.npz')
ensemble = data['ensemble']

H = ea.estimate_hurst(ensemble, T, time_step)
M = eta.estimate_moses(ensemble, min_T, max_T, delta, time_step)
L = eta.estimate_noah(ensemble, M, min_T, max_T, delta, time_step)
J = eta.estimate_joseph(ensemble, min_delta, max_delta, time_step)

summation = M+L+J-1
print(f'Summation: {summation}')
