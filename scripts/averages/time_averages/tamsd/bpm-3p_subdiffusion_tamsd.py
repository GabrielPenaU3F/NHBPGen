import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1/4, 1, 1)

averager = TimeAverager()
T = 10000
time_step = 1
delta_min = 1
delta_max = 100
delta_axis = np.arange(delta_min, delta_max + 1)

tamsd_delta = averager.tamsd(bpm3p, T, delta_min, delta_max, time_step)

log_deltas = np.log(delta_axis)
log_tamsd = np.log(tamsd_delta)

slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_tamsd)

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(delta_axis, tamsd_delta, 'o-', label='TAMSD- Slope = ' + str(slope))
ax.set_xlabel('Time lag (Δ)', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_xlim(auto=True)
ax.set_ylabel('TAMSD', fontsize=11)
ax.yaxis.set_tick_params(labelsize=10)
ax.yaxis.labelpad = 8
ax.set_title('TAMSD (subdiffusion)', fontsize=14)
ax.set_ylim(auto=True)
ax.patch.set_facecolor("#ffffff")
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth(1)
ax.set_facecolor("#ffffff")
ax.legend(fontsize=12)
fig.tight_layout()

plt.show()
