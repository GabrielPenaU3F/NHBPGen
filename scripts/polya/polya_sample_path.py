from domain.polya_process import PolyaProcess

polya = PolyaProcess((1, 0.7))

arrivals = polya.generate_sample_path(10)

print(arrivals)
