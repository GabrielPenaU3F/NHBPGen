import numpy as np
from matplotlib import pyplot as plt

from domain.calculators.ensemble_averager import EnsembleAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(0.25, 1, 1)

averager = EnsembleAverager()
N = 1000
T = 1000
step_length = 1
average = averager.average(bpm3p, N, T, step_length)

t = range(0, T)

plt.plot(t, average)
plt.show()
