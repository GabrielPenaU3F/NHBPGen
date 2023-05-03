from domain.processes.bpm_process import BPMProcess

bpm = BPMProcess((1, 0.8))

arrivals = bpm.generate_sample_path(20)

print(arrivals)
