import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queue_system import QueueSystem

# Classify with K-Means using the full signals: jump time - size

arrivals_process = PolyaProcess(1, 0.9)
service_process = PoissonProcess(0.8)
serv_number = 1

queues = []
fig, ax = plt.subplots(5, 5)
for i in range(25):
    q = QueueSystem(arrivals_process, service_process, serv_number)
    q.simulate_queue(1000)
    q_states = q.export_states()
    queues.append(list(tuple(zip(q_states[0], q_states[1]))))
    ax[int(i / 5)][i % 5].step(q_states[0], q_states[1])

# Preprocessing for classification

# Determine the maximum length among all time series
max_length = max(len(seq) for seq in queues)

# Pad the time series to the maximum length
padded_sequences = []
for seq in queues:
    print(len(seq))
    padded_seq = [tuple((0, 0))] * (max_length - len(seq)) + seq
    padded_sequences.append(padded_seq)

# Convert to NumPy array
X = np.array(padded_sequences)
# Separate the time index and signal value into separate arrays
time_indexes = np.array([[pair[0] for pair in sample] for sample in X])
signal_values = np.array([[pair[1] for pair in sample] for sample in X])
# Reshape the time index and signal value arrays to 2D
num_samples, max_length = time_indexes.shape
reshaped_time_indexes = time_indexes.reshape(num_samples, max_length)
reshaped_signal_values = signal_values.reshape(num_samples, max_length)

# Concatenate the time index and signal value arrays
reshaped_X = np.concatenate((reshaped_time_indexes, reshaped_signal_values), axis=1)

kmeans = KMeans(n_clusters=2)
kmeans.fit(reshaped_X)

print(kmeans.labels_)

plt.show()
