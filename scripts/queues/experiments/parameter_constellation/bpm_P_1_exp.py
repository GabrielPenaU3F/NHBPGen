import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from domain.processes.bpm_process import BPMProcess
from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queues.queue_system import QueueSystem
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci


n_queues = 1000
gamma, beta = 2, 1.8
arrival_process = BPMProcess(gamma, beta)
service_process = PolyaProcess(gamma, beta)
n_servers = 1
time = 100

sizes = []
fig, ax = plt.subplots(10, 10)
for i in range(n_queues):
    q = QueueSystem(arrival_process, service_process, n_servers)
    q.simulate_queue(time)
    q_states = q.export_states()
    sizes.append(q.get_final_size())
    X = np.array(q_states[0])
    y = np.array(q_states[1])
    print(i)

    # These are used to plot if and only if 100 queues are simulated
    # t = np.linspace(0, q_states[0][-1], 100)
    # ax[int(i / 10)][i % 10].set_xticks([])
    # ax[int(i / 10)][i % 10].step(q_states[0], q_states[1])

confidence = .95
threshold = 20
classif_labels = np.array(list(map(lambda x: 1 if x <= threshold else 0, sizes)))
n = len(classif_labels)

proportion_of_stable_queues = classif_labels.mean()
ic_lower_ones, ic_upper_ones = calculate_proportion_ci(proportion_of_stable_queues, n, confidence)
print('Proportion of stable queues: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ic_lower_ones) + ', ' + str(ic_upper_ones) + ')')

# plt.show()
