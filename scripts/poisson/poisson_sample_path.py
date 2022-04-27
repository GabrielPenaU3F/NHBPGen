from src.domain.poisson_process import PoissonProcess

poisson = PoissonProcess(0.7)

arrivals = poisson.generate_sample_path(10)

print(arrivals
      )