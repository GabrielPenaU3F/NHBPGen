import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 200
delta = 2
min_delta = 2
max_delta = 50

data = np.load('../../../test/test_data/test_ensemble_hyperballistic_3.npz')
ensemble = data['ensemble']

H = ea.estimate_hurst(ensemble)
M = eta.estimate_moses(ensemble, T, delta, t_asymp=(150, 190))
L = eta.estimate_noah(ensemble, M, T, delta, t_asymp=(150, 190))
J = eta.estimate_joseph(ensemble, min_delta, max_delta)

summation = M+L+J-1
print(f'Summation: {summation}')
