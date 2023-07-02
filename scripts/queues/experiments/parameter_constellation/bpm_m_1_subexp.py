import numpy as np
from matplotlib import pyplot as plt

from domain.processes.bpm_process import BPMProcess
from domain.processes.poisson_process import PoissonProcess
from domain.queues.queue_system import QueueSystem
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci


n_queues = 1000
gamma, beta = 1.5, 2
arrival_process = BPMProcess(gamma, beta)
service_process = PoissonProcess(1)
n_servers = 1
time = 1000

sizes = []
fig, ax = plt.subplots(10, 10)
for i in range(n_queues):
    q = QueueSystem(arrival_process, service_process, n_servers)
    q.simulate_queue(time)
    sizes.append(q.get_final_size())
    q_states = q.export_states()
    x = np.array(q_states[0])
    y = np.array(q_states[1])
    t = np.linspace(0, q_states[0][-1], 100)
    # These are used to plot if and only if 100 queues are simulated
    # ax[int(i / 10)][i % 10].set_xticks([])
    # ax[int(i / 10)][i % 10].step(q_states[0], q_states[1])

confidence = .95
classif_labels = np.array(list(map(lambda x: 1 if x <= 25 else 0, sizes)))
n = len(classif_labels)

proportion_of_stable_queues = classif_labels.mean()
ic_lower_ones, ic_upper_ones = calculate_proportion_ci(proportion_of_stable_queues, n, confidence)
print('Proportion of stable queues: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ic_lower_ones) + ', ' + str(ic_upper_ones) + ')')

# plt.show()
