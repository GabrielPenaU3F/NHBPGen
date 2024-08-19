from domain.processes.yule_process import YuleProcess
from domain.sampler.sampler import Sampler

yule = YuleProcess(0.1, 1)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(yule, 1000, path_type='arrivals')
