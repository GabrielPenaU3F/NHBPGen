import numpy as np

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

eta = EnsembleTimeAverager()
N = 100
T = 10**5
delta = 90000
min_delta = 10
time_step = 1

data = np.load('../../../test/test_data/test_ensemble_normaldif.npz')
ensemble = data.get('ensemble')[0:N]

J = eta.estimate_joseph(ensemble, min_delta, delta, time_step)
