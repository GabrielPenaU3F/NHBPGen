import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queue_system import QueueSystem

# Classify with K-Means using R^2 coefficients from a linear regression on the sample path

arrivals_process = PolyaProcess(1, 0.9)
service_process = PoissonProcess(0.8)
serv_number = 1

rsqs = []
fig, ax = plt.subplots(5, 5)
for i in range(25):
    q = QueueSystem(arrivals_process, service_process, serv_number)
    q.simulate_queue(1000)
    q_states = q.export_states()
    X = np.array(q_states[0]).reshape((-1, 1))
    y = np.array(q_states[1])
    lr = LinearRegression()
    lr.fit(X, y)
    rsqs.append(lr.score(X, y))
    slope, intercept = lr.coef_, lr.intercept_
    t = np.linspace(0, q_states[0][-1], 10000)
    line = intercept + slope * t
    ax[int(i / 5)][i % 5].step(q_states[0], q_states[1])
    ax[int(i / 5)][i % 5].plot(t, line)

kmeans = KMeans(n_clusters=2)
kmeans.fit(np.array(rsqs).reshape(-1, 1))

print(kmeans.labels_)

plt.show()
