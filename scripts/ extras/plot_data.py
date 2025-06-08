from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.processes.bpm_3p_process import BPM3pProcess

data = DataManager.load_data('data\\120_agg_1d.csv')
# data = DataManager.load_data('data\\190_agg_1d.csv')
# data = DataManager.load_data('data\\1774_agg_1d.csv')
t = data['t_idx'][:]
y = data['cumul_packets'][:]

# bpm3p = BPM3pProcess(1, 1e6, 1.4, initial_state=int(0.4e8))

plt.plot(t, y)
# plt.plot(t[48:], bpm3p.mean_value(t[48:]))
plt.show()

