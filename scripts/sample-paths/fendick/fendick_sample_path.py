from domain.processes.fendick_process import FendickProcess
from domain.sampler.sampler import Sampler

fendick = FendickProcess(1.2, 1, 0.5)
sampler = Sampler()

arrivals = sampler.simulate_sample_path(fendick, 1000, path_type='arrivals')
