import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from domain.queue_system import QueueSystem


def generate_dataset_of_queues(n_queues, arrivals_process, service_process, n_servers, time):
    queues = []
    rsqs = []
    for i in range(n_queues):
        q = QueueSystem(arrivals_process, service_process, n_servers)
        q.simulate_queue(time)
        q_states = q.export_states()
        X = np.array(q_states[0]).reshape((-1, 1))
        y = np.array(q_states[1])
        lr = LinearRegression()
        lr.fit(X, y)
        queues.append(q_states)
        rsqs.append(lr.score(X, y))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(np.array(rsqs).reshape(-1, 1))
    return queues, np.array(kmeans.labels_)

