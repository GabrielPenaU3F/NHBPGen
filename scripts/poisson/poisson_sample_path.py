from domain.processes.poisson_process import PoissonProcess

poisson = PoissonProcess(1.2)

arrivals = poisson.generate_sample_path(100)
