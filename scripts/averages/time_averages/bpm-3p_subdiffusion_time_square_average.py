import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(1/4, 1, 1)

averager = TimeAverager()
N = 100
T = 1000
delta = 10
time_step = 1

t = np.arange(delta, T + time_step, time_step)
sample_path = Sampler().simulate_sample_path(bpm3p, T, path_type='observations', time_step=time_step, plot=False)
avg_t = averager.time_average_as_function_of_t(sample_path, T, delta, time_step, average_type='sq')

slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(avg_t))

fig, ax = plt.subplots(figsize=(8, 5))

plt.loglog(t, avg_t, 'o-', label='Square velocity average - Slope = ' + str(slope))


ax.set_title('Square velocity time average (subdiffusion)', fontsize=14)
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
ax.legend(fontsize=12)
fig.tight_layout()

plt.show()

