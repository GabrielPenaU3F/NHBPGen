import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

# # CESNET 120
# gamma, l, M = 3, 52, -2560

# # CESNET 190
# gamma, l, M = 0.755, 117, -4295

# CESNET 1774
gamma, l, M = 1.74, 79, -3391

model = PenaSigmoid3pProcess(gamma, l, M)
t_max = 10000
t0 = 200
t = np.arange(t0, t_max)
path = model.mean_value(t)
t_asymp_1 = 4000
t_asymp_2 = 10000
t_asymp_start = t_asymp_1 - t0
t_asymp_end = t_asymp_2 - t0

t_log = np.log(t)
x_by_t_log = np.log(path / t)

slope, intercept, r_value, p_value, std_err = linregress(
    t_log[t_asymp_start:t_asymp_end], x_by_t_log[t_asymp_start:t_asymp_end])

M = slope + 1/2
print(f'Moses exponent: M = {M:.4f}')

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(t_log, x_by_t_log, label=r'M(t)/t', linestyle='-.', linewidth=1.8, color='#00B3EB')
ax.plot(t_log, slope * t_log + intercept, label='Asymptotic line', linewidth=1.5, color='red')
ax.set_xlabel('log(Time)', fontsize=14)
ax.set_ylabel('log(Abs. vel.)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.axvline(t_log[t_asymp_start - 1], linestyle='--')
ax.axvline(t_log[t_asymp_end - 1], linestyle='--')
ax.legend(fontsize=14, loc='lower right')
fig.tight_layout()
plt.show()
