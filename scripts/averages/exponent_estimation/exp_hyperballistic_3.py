import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 200
delta = 50
min_delta = 2
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_hyperballistic_3.npz')
ensemble = data['ensemble']

H = ea.estimate_hurst(ensemble, T, time_step)
M = eta.estimate_moses(ensemble, T, delta, time_step)
L = eta.estimate_noah(ensemble, M, T, delta, time_step)
J = eta.estimate_joseph(ensemble, min_delta, delta, time_step)

summation = M+L+J-1
print(f'Summation: {summation}')