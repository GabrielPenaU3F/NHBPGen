from domain.processes.fendick_process import FendickProcess
from domain.sampler import Sampler

fendick = FendickProcess(1.2, 1, 0.5)
sampler = Sampler()

arrivals = sampler.generate_arrivals_sample_path(fendick, 1000)
