from domain.processes.yule_process import YuleProcess

yule = YuleProcess(0.5, 1)

arrivals = yule.generate_sample_path(10)
