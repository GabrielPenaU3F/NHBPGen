from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(5/4, 1, 1)

averager = EnsembleTimeAverager()
N = 100
min_delta = 5
max_delta = 50
T = 500
time_step = 1

averager.estimate_joseph(bpm3p, N, T, min_delta, max_delta, time_step)