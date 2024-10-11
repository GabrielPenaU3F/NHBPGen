import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager


ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 10**6
delta = 10000
min_delta = 100
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_subdif.npz')
ensemble = data.get('ensemble')

H = ea.estimate_hurst(ensemble, T, time_step)
M = eta.estimate_moses(ensemble, T, delta, time_step)
L = eta.estimate_noah(ensemble, M, T, delta, time_step)
J = eta.estimate_joseph(ensemble, min_delta, delta, time_step)

summation = M+L+J-1
print(f'Summation: {summation}')
