import numpy as np

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queues.queue_system import QueueSystem
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci

# Fix service process as a Poisson process
# Consider BPM arrivals with gamma/rho = 1 and a moderate gamma

# DISCARDED EXPERIMENT

arrivals_process = PolyaProcess(0.8, 1)
service_process = PoissonProcess(1)
serv_number = 1

sizes = []
for i in range(100):
    q = QueueSystem(arrivals_process, service_process, serv_number)
    q.simulate_queue(1000)
    sizes.append(q.get_final_size())

stable_unstable_queues = list(map(lambda x: x <= 100, sizes))
stable_unstable_queues = np.array(list(map(int, stable_unstable_queues)))
proportion_of_stable_queues = np.mean(stable_unstable_queues)
n = len(stable_unstable_queues)
ci_lower, ci_upper = calculate_proportion_ci(proportion_of_stable_queues, n, .95)
print('Proportion of stable queues: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ci_lower) + ', ' + str(ci_upper) + ')')

