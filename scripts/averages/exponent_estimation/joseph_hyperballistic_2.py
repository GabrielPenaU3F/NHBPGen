from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(3/2, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
N = 100
min_delta = 5
max_delta = 50
T = 1000
time_step = 1

ensemble = sampler.generate_ensemble(bpm3p, N, T, path_type='observations', time_step=time_step)
averager.estimate_joseph(ensemble, min_delta, max_delta, time_step)