from domain.processes.bpm_process import BPMProcess

bpm = BPMProcess(0.5, 1)

arrivals = bpm.generate_sample_path(1000)
