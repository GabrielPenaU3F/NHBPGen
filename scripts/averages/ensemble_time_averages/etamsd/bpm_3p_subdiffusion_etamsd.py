import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.sampler.sampler import Sampler

bpm3p = BPM3pProcess(0.25, 1, 1)
sampler = Sampler()
averager = EnsembleTimeAverager()
T = 1000
N = 100
time_step = 1
delta_min = 1
delta_max = 100
delta_axis = np.arange(delta_min, delta_max + 1)

ensemble = sampler.generate_ensemble(bpm3p, N, T, path_type='observations', time_step=time_step)
etamsd_delta = averager.etamsd(ensemble, delta_min, delta_max, time_step)

log_deltas = np.log(delta_axis)
log_etamsd = np.log(etamsd_delta)

slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_etamsd)

fig, ax = plt.subplots(figsize=(8, 5))

plt.loglog(delta_axis, etamsd_delta, 'o-', label='ETAMSD - Slope = ' + str(slope))
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
ax.legend(fontsize=12)
fig.tight_layout()

plt.show()
