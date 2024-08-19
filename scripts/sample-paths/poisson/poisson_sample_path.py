from domain.processes.poisson_process import PoissonProcess
from domain.sampler.sampler import Sampler

poisson = PoissonProcess(1.2)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(poisson, 1000, path_type='arrivals')
