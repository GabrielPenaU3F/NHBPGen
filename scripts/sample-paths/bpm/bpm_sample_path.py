from domain.processes.bpm_process import BPMProcess
from domain.sampler import Sampler

bpm = BPMProcess(0.5, 1)

sampler = Sampler()
arrivals = sampler.generate_arrivals_sample_path(bpm, 1000)
