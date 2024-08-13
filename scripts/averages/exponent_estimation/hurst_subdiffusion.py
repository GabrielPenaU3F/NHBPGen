from domain.averagers.ensemble_averager import EnsembleAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1/4, 1, 1)

averager = EnsembleAverager()
N = 100
T = 10000
time_step = 1

averager.estimate_hurst(bpm3p, N, T, time_step)
