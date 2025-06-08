import numpy as np
from matplotlib import pyplot as plt

from domain.data_management.data_manager import DataManager
from domain.processes.bpm_3p_process import BPM3pProcess
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess
from domain.processes.pena_sigmoid_process import PenaSigmoidProcess

t = np.arange(1, 100)

psig3p = PenaSigmoidProcess(1.74, 1, 0, 79, -3391)
y = psig3p.mean_value(t)

plt.plot(t, y)
# plt.plot(t[48:], bpm3p.mean_value(t[48:]))
plt.show()

