import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler import Sampler

bpm3p = BPM3pProcess(0.25, 1, 1)

averager = TimeAverager()
N = 100
max_T = 1000
time_step = 1
delta = 10

vel_avgs = []
t_axis = np.arange(100, max_T, 1)
states_sample_path = Sampler().generate_observations_sample_path(bpm3p, max_T, time_step, plot=False)
for t in t_axis:
    vel_avg = averager.average(states_sample_path, t, time_step, average_type='abs-vel')
    vel_avgs.append(vel_avg)

slope, intercept, r_value, p_value, std_err = linregress(np.log(t_axis), np.log(vel_avgs))

fig, ax = plt.subplots(figsize=(8, 5))

plt.loglog(t_axis, vel_avgs, 'o-', label='Absolute velocity average - Slope = ' + str(slope))


ax.set_title('Absolute velocity time average (subdiffusion)', fontsize=14)
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

