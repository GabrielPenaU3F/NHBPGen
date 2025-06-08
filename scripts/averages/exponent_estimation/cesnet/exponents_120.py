import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from domain.averagers.time_averager import TimeAverager
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

gamma, l, M = 3, 52, -2560
model = PenaSigmoid3pProcess(gamma, l, M)
time_averager = TimeAverager()

delta = 100
min_delta = 10
max_delta = 1000
t_max = 100000
t = np.arange(0, t_max)
t0 = 80000
t_asymp = t[t0:]
max_T = t_max - t0
delta_axis = np.arange(min_delta, max_delta + 1, 1)

path = model.mean_value(t_asymp)
var = model.variance(t_asymp)
second_moment = var + path**2

avg_t_abs = time_averager.time_average_as_function_of_t(path, delta, average_type='abs')
avg_t_sq = time_averager.time_average_as_function_of_t(path, delta, average_type='sq')
tamsd = time_averager.tamsd(path, min_delta, max_delta)

t_avg = np.arange(delta, max_T + 1, delta)
t_avg_log = np.log(t_avg)
delta_log = np.log(delta_axis)
t_log = np.log(t_asymp)

avg_abs_log = np.log(avg_t_abs)
avg_sq_log = np.log(avg_t_sq)
tamsd_log = np.log(tamsd)
m2_log = np.log(second_moment)

slope_abs, intercept_abs, r_value, p_value, std_err = linregress(t_avg_log, avg_abs_log)
slope_sq, intercept_sq, r_value, p_value, std_err = linregress(t_avg_log, avg_sq_log)
slope_tamsd, intercept_tamsd, r_value, p_value, std_err = linregress(delta_log, tamsd_log)
slope_m2, intercept_m2, r_value, p_value, std_err = linregress(t_log, m2_log)

# This Moses measurement is completely erroneous. The correct theoretical value is -1/2
M = slope_abs + 1 / 2
L = (slope_sq - 2 * M + 2) / 2
J = slope_tamsd / 2
H = slope_m2 / 2

print(f'Moses exponent: M={M:.4f}')
print(f'Noah exponent: L={L:.4f}')
print(f'Joseph exponent: J={J:.4f}')
print(f'Hurst exponent: H={H:.4f}')

print(f'M + L + J - 1 = {M + L + J - 1}')

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
ax_1, ax_2, ax_3 = axes

ax_1.plot(t_avg_log, avg_abs_log, label=r'Abs. average', linestyle='-.', linewidth=1.8, color='#00B3EB')
ax_1.plot(t_avg_log, slope_abs * t_avg_log + intercept_abs, label='Asymptotic line', linewidth=1.5, color='red')
ax_1.set_xlabel('log(Time)', fontsize=14)
ax_1.set_ylabel('log(Abs. vel.)', fontsize=14)
ax_1.tick_params(axis='both', which='major', labelsize=12)
ax_1.legend(fontsize=14, loc='lower right')

ax_2.plot(t_avg_log, avg_sq_log, label=r'Sq. average', linestyle='-.', linewidth=1.8, color='#00B3EB')
ax_2.plot(t_avg_log, slope_sq * t_avg_log + intercept_sq, label='Asymptotic line', linewidth=1.5, color='red')
ax_2.set_xlabel('log(Time)', fontsize=14)
ax_2.set_ylabel('log(Sq. vel.)', fontsize=14)
ax_2.tick_params(axis='both', which='major', labelsize=12)
ax_2.legend(fontsize=14, loc='lower right')

ax_3.plot(delta_log, tamsd_log, label=r'TAMSD', linestyle='-.', linewidth=1.8, color='#00B3EB')
ax_3.plot(delta_log, slope_tamsd * delta_log + intercept_tamsd, label='Asymptotic line', linewidth=1.5, color='red')
ax_3.set_xlabel('log(Delta)', fontsize=14)
ax_3.set_ylabel('log(TAMSD)', fontsize=14)
ax_3.tick_params(axis='both', which='major', labelsize=12)
ax_3.legend(fontsize=14, loc='lower right')

fig.tight_layout()

fig2, ax = plt.subplots()

ax.plot(t_log, m2_log, label=r'Second moment', linestyle='-.', linewidth=1.8, color='#00B3EB')
ax.plot(t_log, slope_m2 * t_log + intercept_m2, label='Asymptotic line', linewidth=1.5, color='red')
ax.set_xlabel('log(Time)', fontsize=14)
ax.set_ylabel('log(Second moment)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=14, loc='lower right')

fig2.tight_layout()

plt.show()
