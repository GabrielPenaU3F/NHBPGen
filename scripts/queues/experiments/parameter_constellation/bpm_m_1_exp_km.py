import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from domain.processes.bpm_process import BPMProcess
from domain.processes.poisson_process import PoissonProcess
from domain.queues.queue_system import QueueSystem
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci


n_queues = 100
gamma, beta = 4, 2
arrival_process = BPMProcess(gamma, beta)
service_process = PoissonProcess(1)
n_servers = 1
time = 100

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
    ax[int(i / 10)][i % 10].set_xticks([])
    ax[int(i / 10)][i % 10].step(q_states[0], q_states[1])

confidence = .95

n = len(sizes)
kmeans = KMeans(n_clusters=2)
kmeans.fit(np.array(sizes).reshape(-1, 1))
classif_labels = np.array(kmeans.labels_)

proportion_of_stable_queues = classif_labels.mean()
ic_lower_ones, ic_upper_ones = calculate_proportion_ci(proportion_of_stable_queues, n, confidence)
print('Proportion of one queues: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ic_lower_ones) + ', ' + str(ic_upper_ones) + ')')

classif_labels_not = np.logical_not(classif_labels).astype(int)
proportion_of_zero_marked_queues = classif_labels_not.mean()
ic_lower_zeros, ic_upper_zeros = calculate_proportion_ci(proportion_of_zero_marked_queues, n, confidence)
print('Proportion of zero queues: ' + str(proportion_of_zero_marked_queues))
print('Confidence interval: (' + str(ic_lower_zeros) + ', ' + str(ic_upper_zeros) + ')')

print('Labels: ' + str(classif_labels))
print('Queue sizes: ' + str(sizes))
print('Centroids: ' + str(kmeans.cluster_centers_))

plt.show()
