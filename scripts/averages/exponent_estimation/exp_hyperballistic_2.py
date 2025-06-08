import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 1000
delta = 5
min_delta = 10
max_delta = 100

data = np.load('../../../test/test_data/test_ensemble_hyperballistic_2.npz')
ensemble = data['ensemble']

H = ea.estimate_hurst(ensemble)
M = eta.estimate_moses(ensemble, T, delta, t_asymp=(600, 900))
L = eta.estimate_noah(ensemble, M, T, delta, t_asymp=(600, 900))
J = eta.estimate_joseph(ensemble, min_delta, max_delta)

summation = M+L+J-1
print(f'Summation: {summation}')
