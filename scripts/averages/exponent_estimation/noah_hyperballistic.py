from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(5/4, 1, 1)

averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1
moses = 0.75

averager.estimate_noah(bpm3p, moses, N, min_T, max_T, time_step)
