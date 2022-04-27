from domain.bpm_process import BPMProcess

bpm = BPMProcess((1, 2.9))

arrivals = bpm.generate_sample_path(20)

print(arrivals)
