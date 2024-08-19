import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1

t = np.arange(min_T, max_T, time_step)
ensemble = sampler.generate_ensemble(bpm3p, N, max_T, path_type='observations', time_step=time_step)
vel_avgs = averager.average_as_function_of_t(ensemble, min_T, max_T, time_step, average_type='sq-vel')
slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(vel_avgs))
M = 0.503
L = (slope - 2 * M + 2) / 2

fig, ax = plt.subplots(figsize=(8, 5))
plt.loglog(t, vel_avgs, 'o-', label='Square velocity average', color='#3A1078')


ax.set_title('Square velocity ET-average (ballistic diffusion)', fontsize=14)
ax.set_xlabel('Time (t)', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_ylabel('Square velocity', fontsize=11)
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
# x_pos = 0.95
# y_pos = 0.95  # Adjust this value to position the text as needed
# plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(x_pos, y_pos), xycoords='axes fraction', fontsize=11,
#              horizontalalignment='right', verticalalignment='top')
# plt.annotate(f'L = {'%.3f'%(L)}, M = {'%.3f'%(M)}', xy=(x_pos, y_pos - 0.05), xycoords='axes fraction',
#              fontsize=11, horizontalalignment='right', verticalalignment='top')

legend_bbox = legend.get_window_extent()
legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
x_pos = legend_bbox.x0 + 0.04
y_pos = legend_bbox.y0  # Adjust this value to position the text as needed
plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(x_pos, y_pos), xycoords='figure fraction', fontsize=11)
plt.annotate(f'L = {'%.3f'%(L)}, M = {'%.3f'%(M)}', xy=(x_pos, y_pos - 0.05), xycoords='figure fraction', fontsize=11)

fig.tight_layout()

plt.show()

