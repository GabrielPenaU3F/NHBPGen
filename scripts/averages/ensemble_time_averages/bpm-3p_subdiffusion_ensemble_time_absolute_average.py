import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(0.25, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
N = 100
min_T = 100
max_T = 1000
time_step = 1

ensemble = sampler.generate_ensemble(bpm3p, N, max_T, path_type='observations', time_step=time_step)
avgs_t = averager.average_as_function_of_t(ensemble, min_T, max_T, time_step, average_type='abs')
t = np.arange(min_T, max_T, time_step)
slope, intercept, r_value, p_value, std_err = linregress(np.log(t), np.log(avgs_t))

fig, ax = plt.subplots(figsize=(8, 5))
plt.loglog(t, avgs_t, 'o-', label='Absolute velocity average - Slope = ' + str(slope))


ax.set_title('Absolute velocity ET-average (subdiffusion)', fontsize=14)
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

