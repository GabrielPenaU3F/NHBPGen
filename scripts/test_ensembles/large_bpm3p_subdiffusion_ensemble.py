import numpy as np

from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1/4, 1, 1)
sampler = Sampler()
N = 1000
T = 10**6
time_step = 1

ensemble = sampler.generate_ensemble(bpm3p, N, T, path_type='observations', time_step=time_step, dtype='int32')

# Save
np.savez_compressed('../../test/test_data/test_ensemble_subdif.npz', ensemble=ensemble)

# Load
# with np.load('../../test/test_data/test_ensemble_subdif.npz') as loaded_data:
#     data = loaded_data['ensemble']

