from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1.5, 1, 1)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(bpm3p, 1000, path_type='arrivals')
