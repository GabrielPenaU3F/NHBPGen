import numpy as np
from scipy.stats import t

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from scripts.queues.experiments.P_G_n_divergence.classified_dataset_generator import generate_dataset_of_queues

n_queues = 1000
arrival_process = PolyaProcess(1, 0.9)
service_process = PoissonProcess(0.8)
n_servers = 1
time = 2000

# The labels are 0s and 1s, since they're obtained from the R2 of a linear regression
# low R2 values correspond to stable queues, high R2 values correspond to unstable queues,
# and the algorithm is marking them as 0 for unstable, 1 for stable
queues, classif_labels = generate_dataset_of_queues(n_queues, arrival_process, service_process,
                                                    n_servers, time)

proportion_of_stable_queues = classif_labels.mean()
std = classif_labels.std()
n = len(classif_labels)
ddof = n - 1
confidence = .95
t_crit = np.abs(t.ppf((1 - confidence)/2, ddof))

ic_radius = std * t_crit/np.sqrt(n)
ic_lower = proportion_of_stable_queues - ic_radius
ic_upper = proportion_of_stable_queues + ic_radius

print('Probability of a queue being stable: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ic_lower) + ', ' + str(ic_upper) + ')')
