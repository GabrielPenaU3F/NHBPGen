from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1.5, 1, 1)

arrivals = bpm3p.generate_sample_path(1000)
