from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1

ensemble = sampler.generate_ensemble(bpm3p, N, max_T, path_type='observations', time_step=time_step)
averager.estimate_moses(ensemble, min_T, max_T, time_step)
