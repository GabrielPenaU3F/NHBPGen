import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

ea = EnsembleAverager()
eta = EnsembleTimeAverager()
N = 1000
T = 8000
delta = 200
min_delta = 10
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_ballisticdif.npz')
ensemble = data['ensemble'][0:N-1]

H = ea.estimate_hurst(ensemble, T, time_step)
M = eta.estimate_moses(ensemble, T, delta, time_step)
L = eta.estimate_noah(ensemble, M, T, delta, time_step)
J = eta.estimate_joseph(ensemble, min_delta, delta, time_step)
#
# summation = M+L+J-1
# print(f'Summation: {summation}')