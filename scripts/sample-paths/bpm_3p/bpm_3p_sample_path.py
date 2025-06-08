import numpy as np
from matplotlib import pyplot as plt

from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(4, 1, 1)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(bpm3p, 100, path_type='arrivals')

ns = np.arange(1, 1 + len(arrivals), 1)
lambdas = bpm3p.intensity_function(ns, arrivals)

plt.plot(arrivals, lambdas)
plt.show()
