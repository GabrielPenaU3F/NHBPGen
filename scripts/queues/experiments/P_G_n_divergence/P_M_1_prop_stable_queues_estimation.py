import numpy as np
from scipy import stats as st

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from scripts.queues.experiments.P_G_n_divergence.classified_dataset_generator import generate_dataset_of_queues
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci

n_queues = 1000
beta, gamma = 1, 0.8
arrival_process = PolyaProcess(gamma, beta)
service_process = PoissonProcess(1)
n_servers = 1
time = 1000

# The labels are 0s and 1s, since they're obtained from the R2 of a linear regression
# low R2 values correspond to stable queues, high R2 values correspond to unstable queues,

queues, classif_labels = generate_dataset_of_queues(n_queues, arrival_process, service_process,
                                                    n_servers, time, plot=True)
confidence = .95
n = len(classif_labels)

# Queues marked as 1
proportion_of_one_marked_queues = classif_labels.mean()
ic_lower_ones, ic_upper_ones = calculate_proportion_ci(proportion_of_one_marked_queues, n, confidence)
print('Estimated probability of a queue being marked as 1: ' + str(proportion_of_one_marked_queues))
print('Confidence interval: (' + str(ic_lower_ones) + ', ' + str(ic_upper_ones) + ')')

# Queues marked as 0
classif_labels_not = np.logical_not(classif_labels).astype(int)
proportion_of_zero_marked_queues = classif_labels_not.mean()
ic_lower_zeros, ic_upper_zeros = calculate_proportion_ci(proportion_of_zero_marked_queues, n, confidence)
print('Estimated probability of a queue being marked as 0: ' + str(proportion_of_zero_marked_queues))
print('Confidence interval: (' + str(ic_lower_zeros) + ', ' + str(ic_upper_zeros) + ')')

# Test for the probability of queue being stable

shape = beta/gamma
scale = gamma
gamma_rv = st.gamma(shape, scale=scale)
prob_stable = 1 - gamma_rv.sf(1)

print('Fendicks Gamma distribution probability of stability ' + str(prob_stable))
