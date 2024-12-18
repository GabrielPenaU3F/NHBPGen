import numpy as np
from matplotlib import pyplot as plt

from domain.processes.pena_sigmoid_process import PenaSigmoidProcess
from domain.sampler.sampler import Sampler

gamma, beta, m, l, M = 2, 1, 1, 1000, 4
ps = PenaSigmoidProcess(gamma, beta, m, l, M)
sampler = Sampler()

t = np.linspace(1, 2000, 2000)
mv = ps.mean_value(t)
ensemble = sampler.generate_ensemble(ps, 30, 2000, path_type='observations', time_step=1, plot=False)
path = np.mean(ensemble, axis=0)

fig, ax = plt.subplots()
ax.plot(t, mv, label='Mean value')
ax.plot(t, path, label='Sample path')
ax.legend()

plt.show()
