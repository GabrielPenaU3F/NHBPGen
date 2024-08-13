import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1, 1, 1)

averager = EnsembleTimeAverager()
T = 1000
N = 100
time_step = 1
delta_min = 10
delta_max = 100
delta_axis = np.arange(delta_min, delta_max + 1)

etamsd = averager.etamsd(bpm3p, N, T, delta_min, delta_max, time_step)

slope, intercept, r_value, p_value, std_err = linregress(np.log(delta_axis), np.log(etamsd))
J = slope / 2

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(delta_axis, etamsd, 'o-', label='ETAMSD', color='#3A1078')
ax.set_xlabel('Time lag (Δ)', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_xlim(auto=True)
ax.set_ylabel('ETAMSD')
ax.yaxis.set_tick_params(labelsize=10)
ax.yaxis.labelpad = 8
ax.set_ylim(auto=True)
ax.set_title('ETAMSD (ballistic diffusion)', fontsize=14)
ax.patch.set_facecolor("#ffffff")
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(1)
ax.set_facecolor("#ffffff")
legend = ax.legend(fontsize=12, loc='upper left')

# Annotation
legend_bbox = legend.get_window_extent()
legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
y_pos = legend_bbox.y0  # Adjust this value to position the text as needed
plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(0.11, y_pos), xycoords='figure fraction', fontsize=11)
plt.annotate(f'J = {'%.3f'%(J)}', xy=(0.11, y_pos - 0.05), xycoords='figure fraction', fontsize=11)

fig.tight_layout()

plt.show()
