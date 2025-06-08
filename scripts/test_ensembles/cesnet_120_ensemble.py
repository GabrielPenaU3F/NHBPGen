import numpy as np

from domain.data_management.data_manager import DataManager
from domain.fitting.fitter import Fitter
from domain.fitting.fitter_models.pena_sigmoid_3p_fitter_model import PenaSigmoid3pFitterModel
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess
from domain.sampler.sampler import Sampler

data = DataManager.load_data('data\\120_agg_1d.csv')
fitter_model = PenaSigmoid3pFitterModel()
t = data['t_idx'][:]
y = data['cumul_packets'][:]

fit = Fitter.fit_model(data, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
params = fit.get_params()
model = PenaSigmoid3pProcess(*params)

N = 30
T = 300
time_step = 1

print('Generating...')

ensemble = Sampler().generate_ensemble(model, N, T, path_type='observations', time_step=time_step)

# Save
np.savez_compressed('../../data/simulation_data/cesnet_120_1d_ensemble_icdf_sim.npz', ensemble=ensemble)
