from domain.processes.polya_process import PolyaProcess
from domain.sampler.sampler import Sampler

polya = PolyaProcess(1.2, 10)

sampler = Sampler()
arrivals = sampler.simulate_sample_path(polya, 1000, path_type='arrivals')
