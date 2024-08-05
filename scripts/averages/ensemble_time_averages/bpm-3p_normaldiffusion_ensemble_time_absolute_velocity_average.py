import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(0.5, 1, 1)

averager = EnsembleTimeAverager()
window_length = 100
N = 100
T = 1000
time_step = 1
delta_max = 100
delta_axis = np.arange(1, delta_max + 1)
absavg_delta = []

for delta in delta_axis:
    absavg = averager.ensemble_time_absolute_velocity_average(bpm3p, N, T, delta, time_step)
    if absavg == 0:
        absavg = np.min(absavg_delta)
    absavg_delta.append(absavg)

log_deltas = np.log(delta_axis)
log_vel_avg = np.log(absavg_delta)

slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_vel_avg)

fig, ax = plt.subplots(figsize=(8, 5))

plt.loglog(log_deltas, log_vel_avg, 'o-', label='Absolute velocity average - Slope = ' + str(slope))


ax.set_title('Absolute velocity ET-average (Normal diffusion)', fontsize=14)
ax.set_xlabel('Time lag (Δ)', fontsize=11)
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
ax.legend(fontsize=12)
fig.tight_layout()

plt.show()

