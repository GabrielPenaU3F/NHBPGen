import numpy as np
from matplotlib import pyplot as plt

from domain.processes.pena_sigmoid_process import PenaSigmoidProcess
from domain.sampler.sampler import Sampler

ps = PenaSigmoidProcess(2, 1, 1000, 4)
sampler = Sampler()

t = np.linspace(1, 2000, 2000)
mv = ps.mean_value(t)
path = sampler.simulate_sample_path(ps, 2000, path_type='observations', time_step=1, plot=False)

fig, ax = plt.subplots()
ax.plot(t, mv)
ax.plot(t, path)

plt.show()
