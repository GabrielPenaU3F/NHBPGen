import numpy as np

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(0.25, 1, 1)

averager = EnsembleTimeAverager()
T = 10000
N = 100
step_length = 1
delta_max = 1000
delta_axis = np.arange(1, delta_max + 1)
etamsd_delta = []
for delta in delta_axis:
    etamsd = averager.etamsd(bpm3p, N, T, delta, step_length)
    etamsd_delta.append(etamsd)