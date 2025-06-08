import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager


ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 10**6
delta = 1000
min_delta = 1000
max_delta = 10000

data = np.load('../../../test/test_data/test_ensemble_subdif.npz')
ensemble = data.get('ensemble')[:20]

H = ea.estimate_hurst(ensemble)
M = eta.estimate_moses(ensemble, T, delta, t_asymp=(int(6e5), int(1e6)))
L = eta.estimate_noah(ensemble, M, T, delta, t_asymp=(int(6e5), int(1e6)))
J = eta.estimate_joseph(ensemble, min_delta, max_delta)

summation = M+L+J-1
print(f'Summation: {summation}')
