from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.fitting.fitter import Fitter
from domain.fitting.fitter_models.pena_sigmoid_3p_fitter_model import PenaSigmoid3pFitterModel
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

data = DataManager.load_data('..\\data\\1774_agg_1d.csv')
fitter_model = PenaSigmoid3pFitterModel()

t = data['t_idx'][:]
y = data['cumul_packets'][:]

fit = Fitter.fit_model(data, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
params = fit.get_params()
model = PenaSigmoid3pProcess(*params)
y_pred = model.mean_value(t)

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(t, y, label='Real data', linestyle='-.', color='#00B3EB', linewidth=2.5)
ax.plot(t, y_pred, label='Predicted', linestyle='-', color='#EA114F')
ax.set_xlabel('Time (days)', fontsize=18)
ax.set_ylabel('Cumulative Packets', fontsize=18)
ax.set_title('Real data vs model prediction', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.yaxis.get_offset_text().set_fontsize(18)
ax.legend(fontsize=20)

fig.tight_layout()
# fig.savefig('1774_mean_vs_data.pdf')
plt.show()

print('---- Results ----')
print(r'$\gamma$ = ' + f'{params[0]}')
print(f'l = {params[1]}')
print(f'M = {params[2]}')
print(r'$R^2$ = ' + f'{fit.get_rsq():.4f}')

