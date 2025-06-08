import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

gamma, l, M = 1.74, 79, -3391
psig3p = PenaSigmoid3pProcess(gamma, l, M)

t = np.arange(0, 1000)
mv = psig3p.mean_value(t)
var = psig3p.variance(t)
moment_two = var + mv**2

fig, axes = plt.subplots(figsize=(12, 8))

# axes[0].plot(t[:80], mv[:80], label='Mean value')
# axes[0].axvline(56, color='red', linestyle='--')
# axes[0].legend()

log_t = np.log(t)[1:]
log_m2 = np.log(moment_two)[1:]

slope_transient, intercept_transient, r_value, p_value, std_err = linregress(log_t[5:40], log_m2[5:40])
slope_asymp, intercept_asymp, r_value, p_value, std_err = linregress(log_t[200:], log_m2[200:])
H_asymp = slope_asymp / 2
H_transient = slope_transient / 2

print(f'Hurst parameter (transient): H={H_transient:.4f}')
print(f'Hurst parameter (asymptotic): H={H_asymp:.4f}')

axes.plot(log_t, log_m2, label=r'$E[X^2]$ (log)', linestyle='-.', linewidth=2.5, color='#00B3EB')
axes.plot(log_t[10:36], slope_transient * log_t[10:36] + intercept_transient, label='Transient slope', linewidth=2.5, color='#EB0C4C')
axes.plot(log_t[200:], slope_asymp * log_t[200:] + intercept_asymp, label='Asymptotic slope', linewidth=2.5, color='#6B3B49')
axes.set_xlabel('log(Time)', fontsize=18)
axes.set_ylabel('log(Cumulative Packets)', fontsize=18)
axes.set_title('Hurst parameter estimation', fontsize=22)
axes.axvline(np.log(36), color='red', linestyle='--')
axes.tick_params(axis='both', which='major', labelsize=14)
axes.yaxis.get_offset_text().set_fontsize(18)
axes.legend(fontsize=20)

fig.tight_layout()
plt.savefig('1774_hurst_estimation.pdf')
plt.show()
