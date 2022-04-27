from src.domain.poisson_process import PoissonProcess

poisson = PoissonProcess(2.9)

arrivals = poisson.generate_sample_path(10)

print(arrivals)