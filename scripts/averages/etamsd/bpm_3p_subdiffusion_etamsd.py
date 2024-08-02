import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.ensemble_time_averager import EnsembleTimeAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(0.25, 1, 1)

averager = EnsembleTimeAverager()
T = 10000
N = 100
step_length = 1
delta_max = 1000
delta_axis = np.arange(1, delta_max + 1)
etamsd_delta = []
for delta in delta_axis:
    etamsd = averager.etamsd(bpm3p, N, T, delta, step_length)
    etamsd_delta.append(etamsd)

log_deltas = np.log(delta_axis)
log_etamsd = np.log(etamsd_delta)


slope, intercept, r_value, p_value, std_err = linregress(log_deltas, log_etamsd)
alpha = slope
print(f"Estimated scaling exponent α: {alpha}")

fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(delta_axis, tamsd_delta, label='TAMSD')

plt.loglog(delta_axis, etamsd_delta, 'o-', label='ETAMSD')
plt.xlabel('Time lag (Δ)')
plt.ylabel('ETAMSD')
plt.title('Log-Log Plot of ETAMSD')
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