from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.fitting.fitter import Fitter
from domain.fitting.fitter_models.bpm3p_fitter_model import BPM3pFitterModel
from domain.fitting.fitter_models.pena_sigmoid_3p_fitter_model import PenaSigmoid3pFitterModel
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

data_1d = DataManager.load_data('data\\120_agg_1d.csv')
fitter_psig = PenaSigmoid3pFitterModel()
fitter_bpm3p = BPM3pFitterModel()

t_1d = data_1d['t_idx'][:]
y_1d = data_1d['cumul_packets'][:]

fit_psig = Fitter.fit_model(data_1d, fitter_psig, start=1, end=-1, p0=(1, 100, -0.1))
fit_bpm3p = Fitter.fit_model(data_1d, fitter_bpm3p, start=48, end=-1, p0=(10, 1e6, 20), initial_state=int(0.4e8))
params_psig = fit_psig.get_params()
params_bpm3p = fit_bpm3p.get_params()
model_psig = PenaSigmoid3pProcess(*params_psig)
model_bpm3p = BPM3pProcess(*params_bpm3p, initial_state=int(0.4e8))

y_pred_psig = model_psig.mean_value(t_1d)
y_pred_bpm3p = model_bpm3p.mean_value(t_1d[48:])

fig, ax = plt.subplots()
ax.plot(t_1d, y_1d, label='Real data')
ax.plot(t_1d, y_pred_psig, label='Predicted data (3p - Pena Sigmoid)')
ax.plot(t_1d[48:], y_pred_bpm3p, label='Predicted data (3p - BPM)')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Cumulative Packets')
ax.set_title('1 day aggregation')
ax.legend()

print('--- 3p Pena Sigmoid ---')
print(r'$\gamma$' + f' = {params_psig[0]}')
print(r'$l$' + f' = {params_psig[1]}')
print(r'$M$' + f' = {params_psig[2]}')

print('\n')

print('--- 3p BPM ---')
print(r'$\gamma$' + f' = {params_bpm3p[0]}')
print(r'$\beta$' + f' = {params_bpm3p[1]}')
print(r'$\rho$' + f' = {params_bpm3p[2]}')

fig.tight_layout()
plt.show()
