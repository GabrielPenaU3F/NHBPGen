import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

averager = EnsembleTimeAverager()
T = 20000
delta = 10
time_step = 1

t = np.arange(delta, T + time_step, time_step)
ensemble = np.load('../../../test/test_data/test_ensemble_superdif.npz').get('ensemble')
avgs_t = averager.average_as_function_of_t(ensemble, T, delta, time_step, average_type='abs')
log_t = np.log(t)
log_avgs_t = np.log(avgs_t)
slope, intercept, r_value, p_value, std_err = linregress(log_t, log_avgs_t)
y_regresline = np.exp(slope * log_t + intercept)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(t, avgs_t, linestyle='-', label='Absolute velocity average', color='#000681', linewidth=2.5)
ax.loglog(t[10:], y_regresline[10:], linestyle='--', label='Regression line', color='#E71700', linewidth=2.0)

# ax.set_title('Absolute velocity ET-average (superdiffusion)', fontsize=18)
ax.set_xlabel('Time (t)', fontsize=22)
ax.xaxis.set_tick_params(labelsize=18)
ax.xaxis.labelpad = 4
ax.set_ylabel('Absolute velocity', fontsize=22)
ax.yaxis.set_tick_params(labelsize=18)
ax.yaxis.labelpad = 8
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.patch.set_facecolor('white')
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(1)
ax.set_facecolor('white')
legend = ax.legend(fontsize=22, framealpha=1)
# ax.grid(True, which='major', ls='-', linewidth=0.5, alpha=1)

legend_bbox = legend.get_window_extent()
legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
x_pos = legend_bbox.x0 + 0.09
y_pos = legend_bbox.y0  # Adjust this value to position the text as needed

print(slope)
print(r_value**2)
#plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(x_pos, y_pos), xycoords='figure fraction', fontsize=11)
#plt.annotate(f'$R^2 = {'%.3f'%(r_value ** 2)}$', xy=(x_pos, y_pos - 0.05), xycoords='figure fraction', fontsize=11)

fig.tight_layout()

plt.savefig('superdif_moses.eps', format='eps')
plt.show()

