import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(1/4, 1, 1)

averager = TimeAverager()
T = 10000
step_length = 1
delta_max = 1000
delta_axis = np.arange(1, delta_max + 1)
tamsd_delta = []
for delta in delta_axis:
    tamsd = averager.tamsd(bpm3p, T, delta, step_length)
    if tamsd == 0:
        tamsd = np.min(tamsd_delta)
    tamsd_delta.append(tamsd)

log_deltas = np.log(delta_axis)
log_tamsd = np.log(tamsd_delta)

slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_tamsd)

fig, ax = plt.subplots(figsize=(8, 5))

plt.loglog(delta_axis, tamsd_delta, 'o-', label='TAMSD- Slope = ' + str(slope))
plt.xlabel('Time lag (Δ)')
plt.ylabel('TAMSD')
plt.title('Log-Log Plot of TAMSD')
plt.legend()
plt.show()

# ax.set_title('TAMSD (subdiffusion)', fontsize=14)
# ax.set_xlabel('Δ', fontsize=11)
# ax.xaxis.set_tick_params(labelsize=10)
# ax.xaxis.labelpad = 4
# ax.set_ylabel('TAMSD(Δ)', fontsize=11)
# ax.yaxis.set_tick_params(labelsize=10)
# ax.yaxis.labelpad = 8
# ax.set_xlim(auto=True)
# ax.set_ylim(auto=True)
# ax.patch.set_facecolor("#ffffff")
# ax.patch.set_edgecolor('black')
# ax.patch.set_linewidth(1)
# ax.set_facecolor("#ffffff")
# ax.legend(fontsize=12)
# fig.tight_layout()
#
# plt.show()

