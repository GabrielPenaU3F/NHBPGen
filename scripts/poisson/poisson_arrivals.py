from src.domain.poisson_process import PoissonProcess

poisson = PoissonProcess(0.7)

arrivals = poisson.generate_arrivals(10, show=True)

print(arrivals)
