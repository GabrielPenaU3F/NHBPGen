from domain.processes.polya_process import PolyaProcess

polya = PolyaProcess(1.2, 10)

arrivals = polya.generate_sample_path(1000)
