from domain.processes.yule_process import YuleProcess
from domain.sampler import Sampler

yule = YuleProcess(0.1, 1)

sampler = Sampler()
arrivals = sampler.generate_arrivals_sample_path(yule, 10)
