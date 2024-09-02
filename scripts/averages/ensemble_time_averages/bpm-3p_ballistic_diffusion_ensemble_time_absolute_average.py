import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

averager = EnsembleTimeAverager()
N = 1000
T = 8000
delta = 200
time_step = 1

t = np.arange(delta, T + time_step, time_step)
ensemble = np.load('../../../test/test_data/test_ensemble_ballisticdif.npz').get('ensemble')
avgs_t = averager.average_as_function_of_t(ensemble, T, delta, time_step, average_type='abs')
avgs_t = avgs_t/np.max(avgs_t)
slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(avgs_t))
M = slope + 1/2

fig, ax = plt.subplots(figsize=(8, 5))
plt.loglog(t[200:], avgs_t[200:], '-', linewidth=1.5, label='Absolute velocity average', color='#3A1078')


ax.set_title('Absolute velocity ET-average (ballistic diffusion)', fontsize=14)
ax.set_xlabel('Time (t)', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_ylabel('Absolute velocity', fontsize=11)
ax.yaxis.set_tick_params(labelsize=10)
ax.yaxis.labelpad = 8
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.patch.set_facecolor("#ffffff")
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(1)
ax.set_facecolor("#ffffff")
legend = ax.legend(fontsize=12, loc='upper left')

# Annotation
legend_bbox = legend.get_window_extent()
legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
x_pos = legend_bbox.x0 + 0.15
y_pos = legend_bbox.y0  # Adjust this value to position the text as needed
plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(x_pos, y_pos), xycoords='figure fraction', fontsize=11)
plt.annotate(f'M = {'%.3f'%(M)}', xy=(x_pos, y_pos - 0.05), xycoords='figure fraction', fontsize=11)

fig.tight_layout()

plt.show()

