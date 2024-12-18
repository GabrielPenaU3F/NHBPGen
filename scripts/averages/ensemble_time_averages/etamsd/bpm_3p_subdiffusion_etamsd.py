import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager

averager = EnsembleTimeAverager()
time_step = 1
delta_min = 100
delta_max = 10000
delta_axis = np.arange(delta_min, delta_max + 1)

ensemble = np.load('../../../../test/test_data/test_ensemble_subdif.npz').get('ensemble')[:10]
etamsd_delta = averager.etamsd(ensemble, delta_min, delta_max, time_step)

log_deltas = np.log(delta_axis)
log_etamsd = np.log(etamsd_delta)
slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_etamsd)
y_regresline = np.exp(slope * log_deltas + intercept)

J = slope / 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(delta_axis, etamsd_delta, '-', label='ETAMSD', color='#000681')
ax.loglog(delta_axis[10:], y_regresline[10:], label='Regression line', color='#E71700', linewidth=1.0)
ax.set_xlabel('Time lag (Δ)', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_xlim(auto=True)
ax.set_ylabel('ETAMSD')
ax.yaxis.set_tick_params(labelsize=10)
ax.yaxis.labelpad = 8
ax.set_ylim(auto=True)
ax.set_title('ETAMSD (subdiffusion)', fontsize=14)
ax.patch.set_facecolor("#ffffff")
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(1)
ax.set_facecolor("#ffffff")
legend = ax.legend(fontsize=12)
ax.grid(True, which='both', ls='-', linewidth=0.5)

legend_bbox = legend.get_window_extent()
legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
x_pos = legend_bbox.x0
y_pos = legend_bbox.y0  # Adjust this value to position the text as needed
plt.annotate(f'Slope: {'%.3f'%(slope)}', xy=(x_pos, y_pos), xycoords='figure fraction', fontsize=11)
plt.annotate(f'J = {'%.3f'%(J)}', xy=(x_pos, y_pos - 0.05), xycoords='figure fraction', fontsize=11)
plt.annotate(f'$R^2 = {'%.3f'%(r_value ** 2)}$', xy=(x_pos, y_pos - 0.1), xycoords='figure fraction', fontsize=11)

fig.tight_layout()

plt.savefig('subdif_joseph.eps', format='eps')
plt.show()