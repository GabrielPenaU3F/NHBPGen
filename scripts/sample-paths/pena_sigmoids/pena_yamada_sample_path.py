import numpy as np
from matplotlib import pyplot as plt

from domain.processes.pena_yamada_process import PenaYamadaProcess
from domain.sampler.sampler import Sampler

gamma, beta, b = 2, 100, 0.02
ps = PenaYamadaProcess(gamma, beta, b)
sampler = Sampler()

t = np.linspace(1, 1000, 1000)
mv = ps.mean_value(t)
path = sampler.simulate_sample_path(ps, 1000, path_type='observations', time_step=1, plot=False)

fig, ax = plt.subplots()
ax.plot(t, mv, label='Mean value')
ax.plot(t, path, label='Sample path')
ax.legend()

plt.show()
