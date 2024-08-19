from domain.processes.bpm_process import BPMProcess
from domain.sampler.sampler import Sampler

bpm = BPMProcess(0.5, 1)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(bpm, 1000, path_type='arrivals')
