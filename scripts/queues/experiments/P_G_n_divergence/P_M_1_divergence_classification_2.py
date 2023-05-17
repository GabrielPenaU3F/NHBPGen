import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queue_system import QueueSystem

# Classify with K-Means using only the final size of the queues

arrivals_process = PolyaProcess(1, 0.9)
service_process = PoissonProcess(0.8)
serv_number = 1

queues = []
fig, ax = plt.subplots(5, 5)
for i in range(25):
    q = QueueSystem(arrivals_process, service_process, serv_number)
    q.simulate_queue(1000)
    q_states = q.export_states()
    queues.append(q)
    ax[int(i / 5)][i % 5].step(q_states[0], q_states[1])

q_sizes = np.array([q.get_final_size() for q in queues])

kmeans = KMeans(n_clusters=2)
kmeans.fit(q_sizes.reshape(-1, 1))

print(kmeans.labels_)

plt.show()
