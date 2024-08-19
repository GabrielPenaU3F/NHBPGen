from domain.averagers.ensemble_averager import EnsembleAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1/4, 1, 1)
sampler = Sampler()
averager = EnsembleAverager()
N = 100
T = 10000
time_step = 1

ensemble = sampler.generate_ensemble(bpm3p, N, T, path_type='observations', time_step=time_step)
averager.estimate_hurst(ensemble, T, time_step)
