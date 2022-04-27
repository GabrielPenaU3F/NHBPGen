from domain.yule_process import YuleProcess

yule = YuleProcess(1.5, 1)

arrivals = yule.generate_sample_path(10)

print(arrivals)
