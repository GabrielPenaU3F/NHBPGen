from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1/4, 1, 1)

averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1

averager.estimate_moses(bpm3p, N, min_T, max_T, time_step)
