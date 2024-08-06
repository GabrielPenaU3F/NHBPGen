import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1, 1, 1)

averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1
delta = 10

t = np.arange(min_T, max_T, time_step)
vel_avgs = averager.average_as_function_of_t(bpm3p, N, min_T, max_T, time_step, average_type='abs-vel')
slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(vel_avgs))

fig, ax = plt.subplots(figsize=(8, 5))
plt.loglog(t, vel_avgs, 'o-', label='Absolute velocity average - Slope = ' + str(slope))


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
ax.legend(fontsize=12)
fig.tight_layout()

plt.show()

