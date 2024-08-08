import numpy as np
from matplotlib import pyplot as plt

from domain.averagers.ensemble_averager import EnsembleAverager
from domain.processes.bpm_3p_process import BPM3pProcess

bpm3p = BPM3pProcess(0.5, 1, 1)

averager = EnsembleAverager()
N = 1000
T = 1000
time_step = 1
average = averager.average(bpm3p, N, T, time_step)

t = np.linspace(0, T, int(T / time_step))
mv = bpm3p.mean_value(t)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, average, label='Ensemble average (N=1000)')
ax.plot(t, mv, color='red', linestyle='--', label='Mean value')

ax.set_title('Ensemble average (normal diffusion)', fontsize=14)
ax.set_xlabel('t', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.xaxis.labelpad = 4
ax.set_ylabel('X(t)', fontsize=11)
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
