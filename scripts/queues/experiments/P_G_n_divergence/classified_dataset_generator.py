import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from domain.queues.queue_system import QueueSystem


def generate_dataset_of_queues(n_queues, arrivals_process, service_process, n_servers, time, classif='km', plot=False):
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

    labels = []
    if classif == 'km':
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(np.array(rsqs).reshape(-1, 1))
        labels = np.array(kmeans.labels_)

    # If not KMeans, we assume it's a threshold
    else:
        threshold = classif
        for r2 in rsqs:
            if r2 <= threshold:  # Stable queue
                labels.append(1)
            else:
                labels.append(0)  # Unstable queue
        labels = np.array(labels)

    if plot is True:
        fig, ax = plt.subplots()
        ax.step(q_states[0], q_states[1])
        ax.set_title('Queue marked as ' + str(labels[-1]))
        plt.show()

    return queues, labels

