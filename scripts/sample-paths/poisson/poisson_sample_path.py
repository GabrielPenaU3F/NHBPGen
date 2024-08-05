from domain.processes.poisson_process import PoissonProcess
from domain.sampler import Sampler

poisson = PoissonProcess(1.2)

sampler = Sampler()
arrivals = sampler.generate_arrivals_sample_path(poisson, 1000)
