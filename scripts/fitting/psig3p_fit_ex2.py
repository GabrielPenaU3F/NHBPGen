from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.fitting.fitter import Fitter
from domain.fitting.fitter_models.pena_sigmoid_3p_fitter_model import PenaSigmoid3pFitterModel
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

# np.seterr(invalid='raise')

data_10m = DataManager.load_data('data\\190_agg_10m.csv')
data_1h = DataManager.load_data('data\\190_agg_1h.csv')
data_1d = DataManager.load_data('data\\190_agg_1d.csv')
fitter_model = PenaSigmoid3pFitterModel()

t_10m = data_10m['t_idx'][:]
y_10m = data_10m['cumul_packets'][:]
t_1h = data_1h['t_idx'][:]
y_1h = data_1h['cumul_packets'][:]
t_1d = data_1d['t_idx'][:]
y_1d = data_1d['cumul_packets'][:]

# We use 10 for the aggregated data within 10 mins, 60 for the hourly data and 1440 for the daily data

# fit_10m = Fitter.fit_model(data_10m, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
# params_10m = fit_10m.get_params()
# model_10m = PenaSigmoid3pProcess(*params_10m)
#
# fit_1h = Fitter.fit_model(data_1h, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
# params_1h = fit_1h.get_params()
# model_1h = PenaSigmoid3pProcess(*params_1h)

fit_1d = Fitter.fit_model(data_1d, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
params_1d = fit_1d.get_params()
model_1d = PenaSigmoid3pProcess(*params_1d)

# y_pred_10m = model_10m.mean_value(t_10m)
# y_pred_1h = model_1h.mean_value(t_1h)
y_pred_1d = model_1d.mean_value(t_1d)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# axes[0].plot(t_10m, y_10m, label='Real data')
# axes[0].plot(t_10m, y_pred_10m, label='Predicted data')
# axes[0].set_xlabel('Time (min)')
# axes[0].set_ylabel('Cumulative Packets')
# axes[0].set_title('10 min aggregation')
# axes[0].legend()
#
# axes[1].plot(t_1h, y_1h, label='Real data')
# axes[1].plot(t_1h, y_pred_1h, label='Predicted data')
# axes[1].set_xlabel('Time (h)')
# axes[1].set_ylabel('Cumulative Packets')
# axes[1].set_title('1 hour aggregation')
# axes[1].legend()

axes[2].plot(t_1d, y_1d, label='Real data')
axes[2].plot(t_1d, y_pred_1d, label='Predicted data')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Cumulative Packets')
axes[2].set_title('1 day aggregation')
axes[2].legend()

# print('--- Aggregation 10m ---')
# print(r'$\gamma$' + f' = {params_10m[0]}')
# print(r'$l$' + f' = {params_10m[1]}')
# print(r'$M$' + f' = {params_10m[2]}')
#
# print('--- Aggregation 1h ---')
# print(r'$\gamma$' + f' = {params_1h[0]}')
# print(r'$l$' + f' = {params_1h[1]}')
# print(r'$M$' + f' = {params_1h[2]}')

print('--- Aggregation 1d ---')
print(r'$\gamma$' + f' = {params_1d[0]}')
print(r'$l$' + f' = {params_1d[1]}')
print(r'$M$' + f' = {params_1d[2]}')

fig.tight_layout()
plt.show()
