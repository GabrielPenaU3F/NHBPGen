import numpy as np
from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.fitting.fitter import Fitter
from domain.fitting.fitter_models.pena_sigmoid_3p_fitter_model import PenaSigmoid3pFitterModel
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

def calculate_prr(y, y_pred):
    termwise_prr = ( (y_pred - y) / y_pred ) ** 2
    return np.sum(termwise_prr)


data = DataManager.load_data('..\\..\\data\\1774_agg_1d.csv')
fitter_model = PenaSigmoid3pFitterModel()

data1 = data[:80]
data2 = data[:120]
data3 = data[:150]

t = data['t_idx'][25:]
y = data['cumul_packets'][25:].values

fit_1 = Fitter.fit_model(data1, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
fit_2 = Fitter.fit_model(data2, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))
fit_3 = Fitter.fit_model(data3, fitter_model, start=1, end=-1, p0=(1, 100, -0.1))

params_1 = fit_1.get_params()
params_2 = fit_2.get_params()
params_3 = fit_3.get_params()

model_1 = PenaSigmoid3pProcess(*params_1)
model_2 = PenaSigmoid3pProcess(*params_2)
model_3 = PenaSigmoid3pProcess(*params_3)
model = PenaSigmoid3pProcess(1.74, 79, -3391)

y_pred_1 = model_1.mean_value(t)
y_pred_2 = model_2.mean_value(t)
y_pred_3 = model_3.mean_value(t)
y_pred = model.mean_value(t)

prr_1 = calculate_prr(y, y_pred_1)
prr_2 = calculate_prr(y, y_pred_2)
prr_3 = calculate_prr(y, y_pred_3)
prr = calculate_prr(y, y_pred)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(t, y, label='Real data', linestyle='-.', color='#00B3EB', linewidth=2.5)
axes[0].plot(t, y_pred_1, label='Predicted (up to sample 80)', linestyle='-', color='#EA114F')
axes[0].set_xlabel('Time (days)', fontsize=18)
axes[0].set_ylabel('Cumulative Packets', fontsize=18)
axes[0].set_title('Real data vs model prediction', fontsize=22)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].yaxis.get_offset_text().set_fontsize(18)
axes[0].legend(fontsize=20)

axes[1].plot(t, y, label='Real data', linestyle='-.', color='#00B3EB', linewidth=2.5)
axes[1].plot(t, y_pred_2, label='Predicted (up to sample 120)', linestyle='-', color='#EA114F')
axes[1].set_xlabel('Time (days)', fontsize=18)
axes[1].set_ylabel('Cumulative Packets', fontsize=18)
axes[1].set_title('Real data vs model prediction', fontsize=22)
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].yaxis.get_offset_text().set_fontsize(18)
axes[1].legend(fontsize=20)

axes[2].plot(t, y, label='Real data', linestyle='-.', color='#00B3EB', linewidth=2.5)
axes[2].plot(t, y_pred_3, label='Predicted (up to sample 150)', linestyle='-', color='#EA114F')
axes[2].set_xlabel('Time (days)', fontsize=18)
axes[2].set_ylabel('Cumulative Packets', fontsize=18)
axes[2].set_title('Real data vs model prediction', fontsize=22)
axes[2].tick_params(axis='both', which='major', labelsize=14)
axes[2].yaxis.get_offset_text().set_fontsize(18)
axes[2].legend(fontsize=20)

fig.tight_layout()

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(t, y, label='Real data', linestyle='-.', color='#00B3EB')
ax2.plot(t, y_pred, label='Predicted (full trajectory)', color='#EA114F')
ax2.set_xlabel('Time (days)', fontsize=18)
ax2.set_ylabel('Cumulative Packets', fontsize=18)
ax2.set_title('Real data vs model prediction', fontsize=22)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.yaxis.get_offset_text().set_fontsize(18)
ax2.legend(fontsize=20)
fig2.tight_layout()

print('--- PRR ---')
print(f'Up to sample 80: PRR = {prr_1}')
print(f'Up to sample 120: PRR = {prr_2}')
print(f'Up to sample 150: PRR = {prr_3}')
print(f'Full trajectory: PRR = {prr}')

plt.show()
