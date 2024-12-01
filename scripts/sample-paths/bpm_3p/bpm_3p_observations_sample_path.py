import numpy as np
from matplotlib import pyplot as plt

from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1.5, 1, 1)

sampler = Sampler()
ensemble = sampler.generate_ensemble(bpm3p, 100, time=1000, path_type='observations', time_step=10)
ensemble_mean = np.mean(ensemble, axis=0)

t = np.arange(0, 1000, 10)
mv = bpm3p.mean_value(t)
plt.plot(t, ensemble_mean)
plt.plot(t, mv)
plt.show()
