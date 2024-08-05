from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler import Sampler

bpm3p = BPM3pProcess(1.5, 1, 1)

sampler = Sampler()
arrivals = sampler.generate_observations_sample_path(bpm3p, time=1000, time_step=10)
