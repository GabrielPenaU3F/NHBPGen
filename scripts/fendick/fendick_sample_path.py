from domain.processes.fendick_process import FendickProcess

fendick = FendickProcess(1.2, 1, 0.5)

arrivals = fendick.generate_sample_path(1000)
