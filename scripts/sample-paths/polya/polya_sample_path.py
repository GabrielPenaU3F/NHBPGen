from domain.processes.polya_process import PolyaProcess
from domain.sampler import Sampler

polya = PolyaProcess(1.2, 10)

sampler = Sampler()
arrivals = sampler.generate_arrivals_sample_path(polya, 1000)

